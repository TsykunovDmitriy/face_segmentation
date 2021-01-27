from enum import IntEnum


class MaskType(IntEnum):
    #enumerating in order "next contains prev"
    FACE = 1
    HEAD = 2
    ALL = 3

    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('MaskType.fromString value error')
        return r

to_string_dict = { 
    MaskType.FACE : 'face',
    MaskType.HEAD : 'head',
    MaskType.ALL : 'all',
}

from_string_dict = { to_string_dict[x] : x for x in to_string_dict.keys() } 
