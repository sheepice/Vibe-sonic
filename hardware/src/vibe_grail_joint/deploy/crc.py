"""Keep SDK packing and CRC polynomial, eliminating Python per-word conversion.

Each call owns its native buffer, so the DDS callback and writer may calculate
checksums concurrently. Unknown SDK implementations retain their original CRC.
"""
import ctypes
import sys


def optimized_crc():
    from unitree_sdk2py.utils.crc import CRC
    original = CRC()
    required = ('_CRC__Trans','_crc_ctypes','crc_lib','platform')
    if sys.byteorder != 'little' or not all(hasattr(original,name) for name in required):
        return original

    class BufferCRC(type(original)):
        def _CRC__Trans(self,packed):
            # The last word is the transmitted checksum, excluded by the SDK.
            count = len(packed)//4-1
            return (ctypes.c_uint32*count).from_buffer_copy(packed)

        def _crc_ctypes(self,words):
            return self.crc_lib.crc32_core(words,len(words))

    # Bypass the SDK singleton constructor to avoid mutating any shared instance.
    result = object.__new__(BufferCRC)
    result.__dict__.update(original.__dict__)
    # A checksum takes only a few microseconds. Releasing/reacquiring the GIL
    # for this tiny C function lets the 500 Hz receiver and writer stall each
    # other for an entire Python thread quantum. Keep it for this pure C call.
    result.crc_lib = ctypes.PyDLL(original.crc_lib._name)
    result.crc_lib.crc32_core.argtypes = (ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint32)
    result.crc_lib.crc32_core.restype = ctypes.c_uint32
    result.buffer_conversion = True
    return result
