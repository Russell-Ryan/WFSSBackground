from .exceptions import InstrumentNotSupported

class Instrument:

    @property
    def slowaxis(self):
        return 1-self.fastaxis


class NIRCam(Instrument):
    npix = 2040
    refpix = 4
    
    channels = {1: slice(   4,  512, 1),
                2: slice( 512, 1024, 1),
                3: slice(1024, 1536, 1),
                4: slice(1536, 2044, 1)}

    scipix = (slice(4, 2044, 1), slice(4, 2044, 1))
    sciext = ('SCI', 1)
    uncext = ('ERR', 1)
    dqaext = ('DQ', 1)
    
    fastaxis = 1

    def fast(self):
        for pix in self.channels.values():
            yield self.scipix[0], pix, pix.start, pix.stop
    
class NIRISS(Instrument):
    npix = 2040
    refpix = 4
    
    channels = {1: slice(   4,  512, 1),
                2: slice( 512, 1024, 1),
                3: slice(1024, 1536, 1),
                4: slice(1536, 2044, 1)}

    scipix = (slice(4, 2044, 1), slice(4, 2044, 1))
    sciext = ('SCI', 1)
    uncext = ('ERR', 1)
    dqaext = ('DQ', 1)
    
    fastaxis = 0
    
    def fast(self):
        for pix in self.channels.values():
            yield pix, self.scipix[1], pix.start, pix.stop

def get_instrument(arg):
    if isinstance(arg, Instrument):
        return arg
    elif isinstance(arg, str):
        name = arg.lower()
        if name == 'niriss':
            return NIRISS()
        elif name == 'nircam':
            return NIRCam()
        else:
            raise InstrumentNotSupported(f'The instrument {name} is not defined.')
    else:
        raise TypeError(f'Invalid type: {type(arg)}')
