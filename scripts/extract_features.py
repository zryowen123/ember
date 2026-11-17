#!/usr/bin/env python
"""Safely extract EMBER raw features from PE files inside untrusted archives.

Supported archives: zip (ZipCrypto; AES with pyzipper), 7z (py7zr), tar.gz.

Design notes:
- Samples are read into memory only. Nothing is extracted to disk and nothing
  from the archive is ever executed.
- Only files with a valid PE structure are processed; everything else is
  recorded in the skipped log.
- Each sample is parsed in its own child process with CPU/file-size/core
  rlimits, so a malformed PE or a LIEF crash cannot take down the batch.
- The output is a JSONL file whose records are compatible with the EMBER raw
  feature format (and therefore with ``ember.vectorize_subset``).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import stat
import struct
import sys
import tarfile
import time
import zipfile
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

MIB = 1024 * 1024
GIB = 1024 * MIB
DEFAULT_MAX_FILE_SIZE = 20 * MIB
DEFAULT_MAX_TOTAL_BYTES = 10 * GIB
DEFAULT_MAX_MEMBERS = 100_000
DEFAULT_MAX_COMPRESSION_RATIO = 1000.0
DEFAULT_MAX_RECORD_BYTES = 16 * MIB
DEFAULT_TIMEOUT = 30.0
DEFAULT_MEMORY_PER_SAMPLE = 0  # disabled by default; container memory is authoritative
ZIP_AES_COMPRESS_TYPE = 99
SEVEN_Z_MAGIC = b"7z\xbc\xaf\x27\x1c"
ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
GZIP_MAGIC = b"\x1f\x8b"
PE_SIGNATURE = b"PE\0\0"
BATCH_BUDGET_BYTES = 256 * MIB
BATCH_MAX_MEMBERS = 256


class ArchiveError(Exception):
    """Fatal error for the current archive."""


class ArchiveAbort(ArchiveError):
    """The archive hit a configured limit; stop processing it."""


@dataclass(frozen=True)
class Limits:
    max_file_size: int
    max_total_bytes: int
    max_members: int
    max_compression_ratio: float
    max_record_bytes: int
    timeout: float
    cpu_timeout: float
    memory_per_sample: int


@dataclass
class Member:
    """One archive member.

    Exactly one of ``data``, ``skip_reason`` or ``error_reason`` is meaningful.
    """

    name: str
    size: int = 0
    data: Optional[bytes] = None
    skip_reason: Optional[str] = None
    error_reason: Optional[str] = None


@dataclass
class Pending:
    proc: Any
    conn: Any
    digest: bytes
    sha256: str
    archive: str
    member: str
    label: int
    deadline: float


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMGTP]?)B?\s*", value, re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}")
    number = float(match.group(1))
    suffix = match.group(2).upper()
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
        "P": 1024 ** 5,
    }[suffix]
    result = int(number * multiplier)
    if result < 0:
        raise argparse.ArgumentTypeError(f"invalid size: {value!r}")
    return result


def sanitize_name(name: Any, max_len: int = 512) -> str:
    """Make an archive/member name safe for logs and JSON output.

    The result is never used as a filesystem path.
    """
    text = str(name)
    # Replace lone surrogates and undecodable characters from hostile archives.
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    text = "".join(ch if (ord(ch) >= 0x20 and ord(ch) != 0x7F) else "?" for ch in text)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


def is_pe(data: bytes) -> bool:
    """Cheap structural PE check; extensions are deliberately ignored."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return False
    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if e_lfanew < 4 or e_lfanew + 4 > len(data):
        return False
    return data[e_lfanew : e_lfanew + 4] == PE_SIGNATURE


def detect_format(path: str, override: str) -> str:
    if override != "auto":
        return override
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
    except OSError as exc:
        raise ArchiveError(f"cannot open archive: {exc}") from exc
    if head.startswith(ZIP_MAGICS):
        return "zip"
    if head.startswith(SEVEN_Z_MAGIC):
        return "7z"
    if head.startswith(GZIP_MAGIC):
        return "tar.gz"
    lower = path.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".7z"):
        return "7z"
    if lower.endswith((".tar.gz", ".tgz")):
        return "tar.gz"
    raise ArchiveError(f"unrecognized archive format: {path}")


def read_capped(handle: Any, max_bytes: int) -> bytes:
    data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ArchiveAbort(f"member exceeds max_file_size ({max_bytes} bytes)")
    return data


def zip_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def iter_zip(path: str, password: Optional[bytes], limits: Limits) -> Iterator[Member]:
    try:
        with zipfile.ZipFile(path, "r") as probe:
            infos = probe.infolist()
    except (zipfile.BadZipFile, OSError, UnicodeDecodeError) as exc:
        raise ArchiveError(f"bad zip: {type(exc).__name__}: {exc}") from exc

    has_aes = any(info.compress_type == ZIP_AES_COMPRESS_TYPE for info in infos)
    if has_aes:
        try:
            import pyzipper  # type: ignore
        except ImportError as exc:
            raise ArchiveError("AES-encrypted zip requires pyzipper") from exc
        archive = pyzipper.AESZipFile(path, "r")
        if password is not None:
            archive.setpassword(password)
    else:
        archive = zipfile.ZipFile(path, "r")
        if password is not None:
            archive.setpassword(password)

    with archive:
        try:
            infos = archive.infolist()
        except (OSError, UnicodeDecodeError) as exc:
            raise ArchiveError(f"cannot list zip: {type(exc).__name__}: {exc}") from exc
        if len(infos) > limits.max_members:
            raise ArchiveAbort(f"archive has {len(infos)} members (limit {limits.max_members})")

        total = 0
        for info in infos:
            name = sanitize_name(info.filename)
            if info.is_dir():
                continue
            if zip_is_symlink(info):
                yield Member(name, info.file_size, skip_reason="symlink")
                continue
            if info.file_size > limits.max_file_size:
                yield Member(name, info.file_size, skip_reason="too_large")
                continue
            if (
                info.file_size >= MIB
                and info.compress_size > 0
                and info.file_size / info.compress_size > limits.max_compression_ratio
            ):
                yield Member(name, info.file_size, skip_reason="compression_ratio")
                continue
            try:
                with archive.open(info, "r", pwd=password) as handle:
                    data = read_capped(handle, limits.max_file_size)
            except ArchiveAbort:
                yield Member(name, info.file_size, skip_reason="too_large_actual")
                continue
            except (RuntimeError, NotImplementedError, zipfile.BadZipFile, OSError, ValueError) as exc:
                yield Member(name, info.file_size, error_reason=f"{type(exc).__name__}: {exc}")
                continue
            total += len(data)
            if total > limits.max_total_bytes:
                raise ArchiveAbort(f"archive exceeds max_total_bytes ({limits.max_total_bytes})")
            yield Member(name, len(data), data=data)


def iter_tar_gz(path: str, limits: Limits) -> Iterator[Member]:
    try:
        raw = open(path, "rb")
    except OSError as exc:
        raise ArchiveError(f"cannot open archive: {exc}") from exc
    with raw:
        try:
            archive = tarfile.open(fileobj=raw, mode="r|gz")
        except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
            raise ArchiveError(f"bad tar.gz: {type(exc).__name__}: {exc}") from exc
        with archive:
            count = 0
            total = 0
            declared_total = 0
            try:
                for member in archive:
                    count += 1
                    if count > limits.max_members:
                        raise ArchiveAbort(
                            f"archive has more than {limits.max_members} members"
                        )
                    name = sanitize_name(member.name)
                    if member.isdir():
                        continue
                    if member.issym() or member.islnk():
                        yield Member(name, 0, skip_reason="link")
                        continue
                    if not member.isfile():
                        yield Member(name, 0, skip_reason="not_regular_file")
                        continue
                    if member.size > limits.max_file_size:
                        # In streaming mode we cannot cheaply discard a huge member.
                        raise ArchiveAbort(f"declared member too large: {name}")
                    declared_total += member.size
                    if declared_total > limits.max_total_bytes:
                        raise ArchiveAbort(
                            f"declared archive size exceeds max_total_bytes ({limits.max_total_bytes})"
                        )
                    handle = archive.extractfile(member)
                    if handle is None:
                        yield Member(name, member.size, error_reason="no member data")
                        continue
                    try:
                        data = read_capped(handle, limits.max_file_size)
                    except (tarfile.TarError, OSError, EOFError, zlib.error, ArchiveAbort) as exc:
                        raise ArchiveAbort(f"read error at {name}: {exc}") from exc
                    total += len(data)
                    if total > limits.max_total_bytes:
                        raise ArchiveAbort(
                            f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
                        )
                    yield Member(name, len(data), data=data)
            except (tarfile.TarError, OSError, EOFError, zlib.error) as exc:
                raise ArchiveError(f"tar.gz read error: {type(exc).__name__}: {exc}") from exc


def iter_7z(path: str, password: Optional[bytes], limits: Limits) -> Iterator[Member]:
    try:
        import py7zr  # type: ignore
        from py7zr.io import BytesIOFactory  # type: ignore
    except ImportError as exc:
        raise ArchiveError("7z archives require py7zr>=1.1,<2") from exc

    password_str: Optional[str] = None
    if password is not None:
        try:
            password_str = password.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArchiveError("7z passwords must be UTF-8") from exc

    try:
        archive = py7zr.SevenZipFile(path, mode="r", password=password_str)
    except Exception as exc:  # py7zr raises several archive/password exceptions
        raise ArchiveError(f"bad 7z: {type(exc).__name__}: {exc}") from exc

    with archive:
        try:
            infos = archive.list()
        except Exception as exc:
            raise ArchiveError(f"cannot list 7z: {type(exc).__name__}: {exc}") from exc
        if len(infos) > limits.max_members:
            raise ArchiveAbort(f"archive has {len(infos)} members (limit {limits.max_members})")

        total = 0
        batch: List[Any] = []
        batch_bytes = 0
        seen_names: set[str] = set()

        def flush(current: List[Any]) -> Iterator[Member]:
            nonlocal total
            if not current:
                return
            factory = BytesIOFactory(limits.max_file_size + 1)
            try:
                archive.reset()
                archive.extract(
                    targets=[info.filename for info in current],
                    factory=factory,
                )
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                for info in current:
                    yield Member(sanitize_name(info.filename), info.uncompressed, error_reason=reason)
                return
            for info in current:
                name = sanitize_name(info.filename)
                try:
                    product = factory.get(info.filename)
                    product.seek(0)
                    data = product.read(limits.max_file_size + 2)
                except KeyError:
                    yield Member(name, info.uncompressed, error_reason="missing extracted member")
                    continue
                if len(data) > limits.max_file_size:
                    yield Member(name, info.uncompressed, skip_reason="too_large")
                    continue
                total += len(data)
                if total > limits.max_total_bytes:
                    raise ArchiveAbort(
                        f"archive exceeds max_total_bytes ({limits.max_total_bytes})"
                    )
                yield Member(name, len(data), data=data)

        for info in infos:
            name = sanitize_name(info.filename)
            if info.is_directory:
                continue
            if info.is_symlink:
                yield Member(name, 0, skip_reason="symlink")
                continue
            if not info.is_file:
                yield Member(name, 0, skip_reason="not_regular_file")
                continue
            if info.filename in seen_names:
                yield Member(name, info.uncompressed, error_reason="duplicate member name")
                continue
            seen_names.add(info.filename)
            if info.uncompressed > limits.max_file_size:
                yield Member(name, info.uncompressed, skip_reason="too_large")
                continue
            if (
                info.uncompressed >= MIB
                and info.compressed
                and info.uncompressed / info.compressed > limits.max_compression_ratio
            ):
                yield Member(name, info.uncompressed, skip_reason="compression_ratio")
                continue
            if batch and (
                len(batch) >= BATCH_MAX_MEMBERS
                or batch_bytes + info.uncompressed > BATCH_BUDGET_BYTES
            ):
                yield from flush(batch)
                batch = []
                batch_bytes = 0
            batch.append(info)
            batch_bytes += info.uncompressed

        yield from flush(batch)


def iter_benign_dir(root: str, limits: Limits) -> Iterator[Member]:
    total = 0
    count = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            relative = os.path.relpath(full_path, root)
            name = sanitize_name(relative)
            try:
                info = os.lstat(full_path)
            except OSError as exc:
                yield Member(name, 0, error_reason=f"{type(exc).__name__}: {exc}")
                continue
            if not stat.S_ISREG(info.st_mode):
                yield Member(name, info.st_size, skip_reason="not_regular_file")
                continue
            count += 1
            if count > limits.max_members:
                raise ArchiveAbort(f"directory has more than {limits.max_members} files")
            if info.st_size > limits.max_file_size:
                yield Member(name, info.st_size, skip_reason="too_large")
                continue
            try:
                with open(full_path, "rb") as handle:
                    data = read_capped(handle, limits.max_file_size)
            except (OSError, ArchiveAbort) as exc:
                yield Member(name, info.st_size, error_reason=f"{type(exc).__name__}: {exc}")
                continue
            total += len(data)
            if total > limits.max_total_bytes:
                raise ArchiveAbort(f"directory exceeds max_total_bytes ({limits.max_total_bytes})")
            yield Member(name, len(data), data=data)


class OutputSink:
    def __init__(self, path: str, lock: bool = False) -> None:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, mode=0o700, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self.fd = os.open(path, flags, 0o600)
        try:
            os.fchmod(self.fd, 0o600)
        except OSError:
            pass
        if lock:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(self.fd)
                self.fd = -1
                raise SystemExit(f"output is locked by another process: {path} ({exc})") from exc
        self.lines = 0

    def write_line(self, line: str) -> None:
        payload = line.encode("utf-8") + b"\n"
        view = memoryview(payload)
        while view:
            try:
                written = os.write(self.fd, view)
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError(errno.EIO, f"short write to {self.path}")
            view = view[written:]
        self.lines += 1
        if self.lines % 64 == 0:
            os.fsync(self.fd)

    def write_obj(self, obj: Dict[str, Any]) -> None:
        try:
            line = json.dumps(obj, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            line = json.dumps(
                {
                    "error": "serialization_failed",
                    "detail": f"{type(exc).__name__}: {exc}"[:300],
                },
                ensure_ascii=True,
            )
        self.write_line(line)

    def close(self) -> None:
        if self.fd >= 0:
            try:
                os.fsync(self.fd)
            finally:
                os.close(self.fd)
                self.fd = -1


class Sinks:
    def __init__(self, output: str, errors: str, skipped: str, max_record_bytes: int) -> None:
        self.max_record_bytes = max_record_bytes
        self.output = OutputSink(output, lock=True)
        self.errors = OutputSink(errors)
        self.skipped = OutputSink(skipped)

    def write_record(self, record: Dict[str, Any]) -> bool:
        try:
            line = json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return False
        if len(line.encode("utf-8")) > self.max_record_bytes:
            return False
        self.output.write_line(line)
        return True

    def close(self) -> None:
        for sink in (self.output, self.errors, self.skipped):
            sink.close()


def scan_completed(path: str, repair: bool) -> Dict[bytes, Optional[int]]:
    """Return {sha256_bytes: label} for records already in the output file."""
    completed: Dict[bytes, Optional[int]] = {}
    if not os.path.lexists(path):
        return completed
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"output is not a regular file: {path}")
    last_good = 0
    with open(path, "rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                if repair:
                    os.truncate(path, last_good)
                    print(
                        f"WARNING: truncated incomplete trailing line in {path}",
                        file=sys.stderr,
                    )
                    break
                raise SystemExit(
                    f"{path} ends with an incomplete line; rerun with --repair "
                    "or remove the partial line"
                )
            line_start = last_good
            last_good += len(line)
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if repair:
                    os.truncate(path, line_start)
                    print(
                        f"WARNING: truncated invalid line in {path} ({exc})",
                        file=sys.stderr,
                    )
                    break
                raise SystemExit(
                    f"invalid JSON in {path} at byte {line_start}: {exc}; "
                    "rerun with --repair to truncate it"
                ) from exc
            sha = obj.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                continue
            try:
                digest = bytes.fromhex(sha)
            except ValueError:
                continue
            label = obj.get("label")
            completed[digest] = label if isinstance(label, int) else None
    return completed


def load_password(password_file: Optional[str], password_env: Optional[str]) -> Optional[bytes]:
    if password_file:
        try:
            info = os.stat(password_file)
        except OSError as exc:
            raise SystemExit(f"cannot read password file: {exc}") from exc
        if stat.S_IMODE(info.st_mode) & 0o077:
            print(
                "WARNING: password file is readable by group/other; chmod 600 it",
                file=sys.stderr,
            )
        with open(password_file, "rb") as handle:
            password = handle.read().strip(b"\r\n")
        if not password:
            raise SystemExit("password file is empty")
        return password
    if password_env:
        value = os.environ.get(password_env)
        if value is None:
            raise SystemExit(f"environment variable {password_env} is not set")
        return value.encode("utf-8")
    return None


def set_child_limits(limits: Limits) -> None:
    try:
        import resource  # type: ignore
    except ImportError:
        return

    def apply(kind: int, value: int, allow_zero: bool = False) -> None:
        if value < 0 or (value == 0 and not allow_zero):
            return
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass

    if limits.cpu_timeout > 0:
        apply(resource.RLIMIT_CPU, int(limits.cpu_timeout) + 1)
    if limits.memory_per_sample > 0:
        apply(resource.RLIMIT_AS, limits.memory_per_sample)
    apply(resource.RLIMIT_FSIZE, 0, allow_zero=True)
    apply(resource.RLIMIT_CORE, 0, allow_zero=True)
    apply(resource.RLIMIT_NOFILE, 64)


def child_main(conn: Any, bytez: bytes, feature_version: int, limits: Limits) -> None:
    try:
        sys.dont_write_bytecode = True
        set_child_limits(limits)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 1)
            os.dup2(devnull, 2)
            os.close(devnull)
        except OSError:
            pass
        from ember.features import PEFeatureExtractor

        extractor = PEFeatureExtractor(feature_version, print_feature_warning=False)
        features = extractor.raw_features(bytez)
        conn.send(("ok", features))
    except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the batch
        try:
            conn.send(("error", type(exc).__name__, str(exc)[:512]))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


def make_context():  # type: ignore[no-untyped-def]
    import multiprocessing as mp

    try:
        ctx = mp.get_context("forkserver")
        try:
            ctx.set_forkserver_preload(["ember.features"])
        except Exception:
            pass
        return ctx
    except ValueError:
        return mp.get_context("spawn")


def next_timeout(pending: Dict[Any, Pending]) -> float:
    if not pending:
        return 0.0
    return max(0.0, min(item.deadline - time.monotonic() for item in pending.values()))


def terminate_item(item: Pending) -> None:
    try:
        item.proc.terminate()
    except Exception:
        pass
    item.proc.join(1.0)
    if item.proc.is_alive():
        try:
            item.proc.kill()
        except Exception:
            pass
        item.proc.join(1.0)


def terminate_all(
    pending: Dict[Any, Pending], inflight: Optional[Dict[bytes, int]] = None
) -> None:
    for conn, item in list(pending.items()):
        terminate_item(item)
        _close_conn(conn)
        pending.pop(conn, None)
        if inflight is not None:
            inflight.pop(item.digest, None)


def wait_for_result(
    pending: Dict[Any, Pending],
    timeout: Optional[float],
    inflight: Optional[Dict[bytes, int]] = None,
) -> None:
    from multiprocessing.connection import wait as mp_wait

    if not pending:
        return
    ready = mp_wait(list(pending.keys()), timeout=timeout)
    if not ready:
        now = time.monotonic()
        for conn, item in list(pending.items()):
            if now >= item.deadline:
                terminate_item(item)
                _close_conn(conn)
                pending.pop(conn, None)
                if inflight is not None:
                    inflight.pop(item.digest, None)
                _record_failure(item, "timeout", "per-sample wall-clock timeout")
        return

    for conn in ready:
        item = pending.pop(conn, None)
        if item is None:
            continue
        if inflight is not None:
            inflight.pop(item.digest, None)
        result: Any = None
        failure: Optional[Tuple[str, str]] = None
        try:
            result = conn.recv()
        except (EOFError, OSError) as exc:
            failure = ("crash", f"{type(exc).__name__}: {exc}")
        finally:
            _close_conn(conn)
            item.proc.join(0.5)
            if item.proc.is_alive():
                try:
                    item.proc.kill()
                except Exception:
                    pass
                item.proc.join(0.5)
        if failure is not None:
            _record_failure(item, failure[0], failure[1])
        elif not isinstance(result, tuple) or len(result) < 2:
            _record_failure(item, "crash", "invalid result from worker")
        elif result[0] == "ok":
            if isinstance(result[1], dict):
                _record_success(item, result[1])
            else:
                _record_failure(item, "crash", "worker returned a non-dict result")
        else:
            reason = ": ".join(str(part) for part in result[1:]) or "worker error"
            _record_failure(item, "parse_error", reason[:1000])


# The following globals are populated in main() so the supervisor callbacks
# can stay small and easy to reason about.
_SINKS: Optional[Sinks] = None
_SEEN: Optional[Dict[bytes, Optional[int]]] = None
_STATS: Optional[Dict[str, int]] = None
_FEATURE_VERSION = 2


def _close_conn(conn: Any) -> None:
    try:
        conn.close()
    except Exception:
        pass


def _record_success(item: Pending, features: Dict[str, Any]) -> None:
    assert _SINKS is not None and _SEEN is not None and _STATS is not None
    record = dict(features)
    record["sha256"] = item.sha256
    record["label"] = item.label
    record["appeared"] = ""
    record["avclass"] = ""
    record["source"] = {"archive": item.archive, "member": item.member}
    if _SINKS.write_record(record):
        _SEEN[item.digest] = item.label
        _STATS["extracted"] += 1
        if item.label == 0:
            _STATS["label_0"] += 1
        elif item.label == 1:
            _STATS["label_1"] += 1
    else:
        _SINKS.errors.write_obj(
            {
                "sha256": item.sha256,
                "archive": item.archive,
                "member": item.member,
                "error": "record_too_large_or_serialization_failed",
                "max_record_bytes": _SINKS.max_record_bytes,
            }
        )
        _SEEN[item.digest] = item.label
        _STATS["record_errors"] += 1


def _record_failure(item: Pending, kind: str, reason: str) -> None:
    assert _SINKS is not None and _SEEN is not None and _STATS is not None
    _SINKS.errors.write_obj(
        {
            "sha256": item.sha256,
            "archive": item.archive,
            "member": item.member,
            "error": kind,
            "message": reason,
        }
    )
    _SEEN[item.digest] = item.label
    if kind == "timeout":
        _STATS["timeouts"] += 1
    elif kind == "crash":
        _STATS["crashes"] += 1
    else:
        _STATS["parse_errors"] += 1


def process_members(
    members: Iterable[Member],
    archive_name: str,
    label: int,
    ctx: Any,
    limits: Limits,
    workers: int,
    seen: Dict[bytes, Optional[int]],
    stats: Dict[str, int],
    sinks: Sinks,
) -> None:
    pending: Dict[Any, Pending] = {}
    inflight: Dict[bytes, int] = {}
    try:
        for member in members:
            stats["members"] += 1
            if member.skip_reason:
                sinks.skipped.write_obj(
                    {
                        "archive": archive_name,
                        "member": member.name,
                        "size": member.size,
                        "reason": member.skip_reason,
                    }
                )
                stats["skipped"] += 1
                continue
            if member.error_reason or member.data is None:
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "member": member.name,
                        "size": member.size,
                        "error": "member_error",
                        "message": member.error_reason or "no data",
                    }
                )
                stats["parse_errors"] += 1
                continue

            data = member.data
            if not is_pe(data):
                sinks.skipped.write_obj(
                    {
                        "archive": archive_name,
                        "member": member.name,
                        "size": len(data),
                        "reason": "not_pe",
                    }
                )
                stats["skipped_not_pe"] += 1
                continue

            digest = hashlib.sha256(data).digest()
            sha256 = digest.hex()
            if digest in seen or digest in inflight:
                previous = seen.get(digest) if digest in seen else inflight.get(digest)
                if isinstance(previous, int) and previous != label:
                    sinks.errors.write_obj(
                        {
                            "sha256": sha256,
                            "archive": archive_name,
                            "member": member.name,
                            "error": "label_conflict",
                            "message": f"already seen with label {previous}, now {label}",
                        }
                    )
                    stats["label_conflicts"] += 1
                else:
                    stats["duplicates"] += 1
                continue

            while len(pending) >= workers:
                wait_for_result(pending, next_timeout(pending), inflight)

            recv_conn, send_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=child_main,
                args=(send_conn, data, _FEATURE_VERSION, limits),
                daemon=True,
            )
            try:
                proc.start()
            except Exception as exc:  # noqa: BLE001 - spawn failure is per-sample
                _close_conn(recv_conn)
                _close_conn(send_conn)
                sinks.errors.write_obj(
                    {
                        "sha256": sha256,
                        "archive": archive_name,
                        "member": member.name,
                        "error": "spawn_failed",
                        "message": f"{type(exc).__name__}: {exc}"[:500],
                    }
                )
                seen[digest] = label
                stats["crashes"] += 1
                continue
            _close_conn(send_conn)
            pending[recv_conn] = Pending(
                proc=proc,
                conn=recv_conn,
                digest=digest,
                sha256=sha256,
                archive=archive_name,
                member=member.name,
                label=label,
                deadline=time.monotonic() + limits.timeout,
            )
            inflight[digest] = label
            stats["pe"] += 1

        while pending:
            wait_for_result(pending, next_timeout(pending), inflight)
    except ArchiveError:
        # The archive hit a limit after some samples were already queued.
        # Finish those samples before reporting the archive-level error.
        while pending:
            wait_for_result(pending, next_timeout(pending), inflight)
        raise
    finally:
        terminate_all(pending, inflight)


def run(args: argparse.Namespace) -> int:
    global _SINKS, _SEEN, _STATS, _FEATURE_VERSION

    if not args.archive and not args.benign_archive and not args.benign_dir:
        raise SystemExit(
            "provide at least one --archive, --benign-archive, or --benign-dir"
        )

    archive_jobs: List[Tuple[str, int]] = []
    archive_labels: Dict[str, int] = {}

    def add_archive_job(path: str, label: int) -> None:
        key = os.path.abspath(path)
        if key in archive_labels:
            if archive_labels[key] != label:
                raise SystemExit(
                    f"archive specified with conflicting labels: {path} "
                    f"({archive_labels[key]} vs {label})"
                )
            return
        archive_labels[key] = label
        archive_jobs.append((path, label))

    for path in args.archive:
        add_archive_job(path, args.label)
    for path in args.benign_archive:
        add_archive_job(path, 0)

    output = os.path.abspath(args.output)
    errors = os.path.abspath(args.errors) if args.errors else output + ".errors.jsonl"
    skipped = os.path.abspath(args.skipped) if args.skipped else output + ".skipped.jsonl"

    password = load_password(args.password_file, args.password_env)

    if args.resume:
        seen = scan_completed(output, args.repair)
    else:
        seen = {}
        if os.path.exists(output) and os.path.getsize(output) > 0:
            raise SystemExit(
                f"output exists and --no-resume was given: {output}; "
                "remove it or use --resume"
            )

    limits = Limits(
        max_file_size=args.max_file_size,
        max_total_bytes=args.max_total_bytes,
        max_members=args.max_members,
        max_compression_ratio=args.max_compression_ratio,
        max_record_bytes=args.max_record_bytes,
        timeout=args.timeout_per_sample,
        cpu_timeout=args.timeout_per_sample,
        memory_per_sample=args.memory_per_sample,
    )

    sinks = Sinks(output, errors, skipped, args.max_record_bytes)
    stats = {
        "archives": 0,
        "members": 0,
        "pe": 0,
        "extracted": 0,
        "skipped": 0,
        "skipped_not_pe": 0,
        "duplicates": 0,
        "timeouts": 0,
        "crashes": 0,
        "parse_errors": 0,
        "record_errors": 0,
        "label_conflicts": 0,
        "label_0": 0,
        "label_1": 0,
        "archive_errors": 0,
    }
    _SINKS = sinks
    _SEEN = seen
    _STATS = stats
    _FEATURE_VERSION = args.feature_version

    ctx = make_context()
    try:
        for archive_path, archive_label in archive_jobs:
            stats["archives"] += 1
            archive_name = sanitize_name(os.path.basename(archive_path))
            if not os.path.exists(archive_path):
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "error": "missing_archive",
                        "message": archive_path,
                    }
                )
                stats["archive_errors"] += 1
                continue
            try:
                fmt = detect_format(archive_path, args.format)
                if fmt == "zip":
                    members: Iterable[Member] = iter_zip(archive_path, password, limits)
                elif fmt == "tar.gz":
                    members = iter_tar_gz(archive_path, limits)
                elif fmt == "7z":
                    members = iter_7z(archive_path, password, limits)
                else:  # pragma: no cover - argparse restricts this
                    raise ArchiveError(f"unsupported format: {fmt}")
                process_members(
                    members,
                    archive_name,
                    archive_label,
                    ctx,
                    limits,
                    args.workers,
                    seen,
                    stats,
                    sinks,
                )
            except ArchiveError as exc:
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
                if args.verbose:
                    print(f"archive error: {archive_name}: {exc}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 - untrusted archive parsers
                sinks.errors.write_obj(
                    {
                        "archive": archive_name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
                if args.verbose:
                    print(
                        f"unexpected archive error: {archive_name}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )

        for benign_dir in args.benign_dir:
            stats["archives"] += 1
            name = sanitize_name(os.path.basename(os.path.abspath(benign_dir)))
            try:
                process_members(
                    iter_benign_dir(benign_dir, limits),
                    name,
                    0,
                    ctx,
                    limits,
                    args.workers,
                    seen,
                    stats,
                    sinks,
                )
            except ArchiveError as exc:
                sinks.errors.write_obj(
                    {
                        "archive": name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
            except Exception as exc:  # noqa: BLE001 - untrusted filesystem input
                sinks.errors.write_obj(
                    {
                        "archive": name,
                        "error": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                stats["archive_errors"] += 1
    finally:
        sinks.close()

    print(
        "summary: "
        + " ".join(f"{key}={value}" for key, value in stats.items()),
        file=sys.stderr,
    )

    if args.vectorize:
        vectorize_output(args, output)
    return 0


def vectorize_output(args: argparse.Namespace, output: str) -> None:
    import ember
    from ember.features import PEFeatureExtractor

    nrows = 0
    with open(output, "rb") as handle:
        for _ in handle:
            nrows += 1
    if nrows == 0:
        print("nothing to vectorize", file=sys.stderr)
        return
    data_dir = args.vectorize_dir or os.path.dirname(output)
    os.makedirs(data_dir, mode=0o700, exist_ok=True)
    extractor = PEFeatureExtractor(args.feature_version, print_feature_warning=False)
    X_path = os.path.join(data_dir, f"X_{args.split}.dat")
    y_path = os.path.join(data_dir, f"y_{args.split}.dat")
    ember.vectorize_subset(X_path, y_path, [output], extractor, nrows)
    print(
        f"vectorized {nrows} rows x {extractor.dim} dims -> {X_path}, {y_path}",
        file=sys.stderr,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely extract EMBER raw features from PE files in zip/7z/tar.gz archives."
    )
    parser.add_argument("--archive", action="append", default=[], help="archive path (repeatable)")
    parser.add_argument(
        "--benign-archive",
        action="append",
        default=[],
        help="archive path labelled 0 (repeatable; overrides --label)",
    )
    parser.add_argument(
        "--benign-dir",
        action="append",
        default=[],
        help="directory of loose PE files labelled 0 (repeatable)",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=(0, 1),
        default=1,
        help="label applied to --archive entries (default: 1)",
    )
    parser.add_argument("--output", required=True, help="output JSONL path")
    parser.add_argument("--errors", help="error JSONL path (default: <output>.errors.jsonl)")
    parser.add_argument("--skipped", help="skipped JSONL path (default: <output>.skipped.jsonl)")
    parser.add_argument(
        "--format",
        choices=("auto", "zip", "7z", "tar.gz"),
        default="auto",
        help="archive format (default: sniff magic bytes)",
    )
    password_group = parser.add_mutually_exclusive_group()
    password_group.add_argument("--password-file", help="file containing the archive password")
    password_group.add_argument("--password-env", help="environment variable with the password")
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="skip sha256 values already present in the output (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="do not scan the output; requires a fresh/empty output file",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="truncate an incomplete/invalid trailing line during resume",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel extraction workers (default: min(4, cpu count))",
    )
    parser.add_argument(
        "--max-file-size",
        type=parse_size,
        default=DEFAULT_MAX_FILE_SIZE,
        help="maximum uncompressed size per member (default: 20M)",
    )
    parser.add_argument(
        "--max-total-bytes",
        type=parse_size,
        default=DEFAULT_MAX_TOTAL_BYTES,
        help="maximum bytes read per archive/directory (default: 10G)",
    )
    parser.add_argument(
        "--max-members",
        type=int,
        default=DEFAULT_MAX_MEMBERS,
        help="maximum members per archive/directory (default: 100000)",
    )
    parser.add_argument(
        "--max-compression-ratio",
        type=float,
        default=DEFAULT_MAX_COMPRESSION_RATIO,
        help="skip members whose declared expansion ratio exceeds this (default: 1000)",
    )
    parser.add_argument(
        "--max-record-bytes",
        type=parse_size,
        default=DEFAULT_MAX_RECORD_BYTES,
        help="maximum serialized feature record size (default: 16M)",
    )
    parser.add_argument(
        "--timeout-per-sample",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="wall-clock seconds per sample (default: 30)",
    )
    parser.add_argument(
        "--memory-per-sample",
        type=parse_size,
        default=DEFAULT_MEMORY_PER_SAMPLE,
        help="optional RLIMIT_AS per child in bytes (0 disables; default: 0)",
    )
    parser.add_argument(
        "--feature-version", type=int, choices=(1, 2), default=2, help="EMBER feature version"
    )
    parser.add_argument(
        "--vectorize",
        action="store_true",
        help="after extraction, vectorize with ember.vectorize_subset",
    )
    parser.add_argument(
        "--vectorize-dir",
        help="directory for X_<split>.dat/y_<split>.dat (default: output directory)",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="name for vectorized output (default: train)",
    )
    parser.add_argument("--verbose", action="store_true", help="print per-archive errors")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.timeout_per_sample <= 0:
        parser.error("--timeout-per-sample must be > 0")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
