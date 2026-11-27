"""A synthetic PE file for the tests of the Windows bundle import check. No binary is committed."""

from __future__ import annotations


def pe_file(imports: tuple[str, ...] = (), delayed: tuple[str, ...] = ()) -> bytes:
    """A minimal 64-bit PE file whose import tables name *imports* and *delayed*.

    One section holds the descriptors, one empty thunk list per DLL and the names.
    The file has no code: it only has to be a PE that pefile reads.
    """
    dos = b"MZ" + bytes(58) + (0x40).to_bytes(4, "little")
    section_rva, section_raw, headers = 0x1000, 0x200, 0x200
    blob = bytearray()

    def put(data: bytes) -> int:
        offset = len(blob)
        blob.extend(data)
        return section_rva + offset

    def u32(*values: int) -> bytes:
        return b"".join(v.to_bytes(4, "little") for v in values)

    # One thunk list serves every DLL: a single function "f" by name, then the null end.
    hint_name = put(bytes(2) + b"f\0\0")
    blob.extend(bytes(-len(blob) % 8))
    thunks = put(hint_name.to_bytes(8, "little") + bytes(8))
    names = [put(n.encode() + b"\0") for n in (*imports, *delayed)]
    blob.extend(bytes(-len(blob) % 8))
    import_rva = section_rva + len(blob)
    for rva in names[: len(imports)]:
        blob.extend(u32(thunks, 0, 0, rva, thunks))
    blob.extend(bytes(20))
    delay_rva = section_rva + len(blob)
    for rva in names[len(imports) :]:
        blob.extend(u32(1, rva, thunks, thunks, thunks, 0, 0, 0))
    blob.extend(bytes(32))

    directories = bytearray(16 * 8)
    if imports:
        directories[8:16] = u32(import_rva, 20 * (len(imports) + 1))
    if delayed:
        directories[13 * 8 : 14 * 8] = u32(delay_rva, 32 * (len(delayed) + 1))
    optional = (
        (0x20B).to_bytes(2, "little")
        + bytes(22)
        + (0x180000000).to_bytes(8, "little")
        + u32(0x1000, 0x200)  # section and file alignment
        + bytes(16)
        + u32(section_rva + 0x1000, headers)  # size of image, size of headers
        + bytes(4)
        + (3).to_bytes(2, "little")  # console subsystem
        + bytes(2)
        + bytes(32)
        + bytes(4)
        + u32(16)  # data directory count
        + bytes(directories)
    )
    assert len(optional) == 240
    coff = (0x8664).to_bytes(2, "little") + (1).to_bytes(2, "little") + bytes(12)
    coff += (240).to_bytes(2, "little") + (0x2022).to_bytes(2, "little")
    section = b".idata\0\0" + u32(len(blob), section_rva, len(blob), section_raw) + bytes(16)
    header = dos + b"PE\0\0" + coff + optional + section
    return header + bytes(headers - len(header)) + bytes(blob)
