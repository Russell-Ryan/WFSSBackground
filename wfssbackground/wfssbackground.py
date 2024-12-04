import importlib.metadata as im
import os
import shutil
from timeit import default_timer
import warnings

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from skimage import morphology


from .instruments import get_instrument


class WFSSBackground:
    ALPHATOL = 0.0
    BETATOL = -2.0
    GAMMATOL = -2.0
    SUFFIX = 'skysub'

    def __init__(self, scifile, objfile=None, skyfile=None, pflfile=None,
                 bitmask=65535):
        self.bitmask = bitmask
        if scifile is not None:
            self.load_data(scifile, objfile, skyfile, pflfile)
        self.results = {}
        
    def load_data(self, scifile, objfile, skyfile, pflfile):
        ''' load the data '''
        
        self.scifile = scifile
        
        with fits.open(self.scifile, mode='readonly') as hdul:
            h0 = hdul[0].header
            self.ins = get_instrument(h0['INSTRUME'])
            self.kern = self.ins.get_kernel(h0)
            
            # read the images
            self.sci = hdul[self.ins.sciext].data
            self.unc = hdul[self.ins.uncext].data
            dqa = hdul[self.ins.dqaext].data

        # find the object mask?
        if isinstance(objfile, str):
            self.obj = fits.getdata(objfile)
        else:
            self.obj = np.zeros_like(self.sci, dtype=bool)
                    
        # read the sky image?
        self.sky, self.skyfile = self.load_reffile(skyfile, h0, 'R_WFSSBCK')

        # load the PFL image
        self.pfl, self.pflfile = self.load_reffile(pflfile, h0, 'R_FLAT')
        
        # find the nans
        nan = np.isnan(self.sci) | np.isnan(self.unc) | np.isnan(self.sky)

        # find the gpx
        gpx = (np.bitwise_and(dqa, self.bitmask) == 0) & np.logical_not(nan)
        
        # set the weights
        self.wht = gpx/np.maximum(self.unc, 1e-10)**2

        # set the nans
        self.nanxy = np.where(nan)
        self.sci[self.nanxy] = 0.
        self.wht[self.nanxy] = 0.
        self.sky[self.nanxy] = 0.

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, obj):
        self._obj = obj.astype(bool)
        

    @staticmethod
    def load_reffile(filename, h0, key):
        ''' get the reference file data '''


        locfile = None

        if filename is None:
            if key in h0:
                filename = h0[key]
                if filename == 'N/A':
                    warnings.warn(f'{key} is not set in header.')
                    return None, None
                
                tel = h0['TELESCOP'].lower()
                ins = h0['INSTRUME'].lower()
                if (tel != 'jwst') or (ins not in ('niriss', 'nircam')):
                    warnings.warn('Only JWST/NIRCam or JWST/NIRISS supported.')
                    return None, None
            
                basename = os.path.basename(filename)
                locpath = (os.environ['CRDS_PATH'], 'references', tel, ins)
                locfile = os.path.join(*locpath)+os.sep+basename

                if not os.path.exists(locfile):
                    try:
                        # try to download the file
                        url = 'https://jwst-crds.stsci.edu/browse/'
                        remfile = url+basename
                        r = requests.get(remfile, stream=True)
                        
                        r.raise_for_status()                    
                        
                        with open(locfile, "wb") as fobj:
                            for chunk in r.iter_content(chunk_size=1024000):
                                fobj.write(chunk)
                    except BaseException:
                        pass
                else:
                    pass

            else:
                pass
        elif isinstance(filename, str) and os.path.exists(filename):
            locfile = filename

        # now read the image
        if locfile:
            data = fits.getdata(locfile)
        else:
            data = None

            
        return data, locfile

            
        
    def fit_model(self, fit_oof=True, fit_slow=True, combine=True, repval=0.,
                  verbose=True, write=False, outfile=None):
        ''' do the model fitting '''
        wht = np.logical_not(self.obj)*self.wht

        if fit_oof:
            # this is the fancy fitting
            nalpha = 1
            nbeta = len(self.ins.channels)*self.ins.npix
            if fit_slow:
                ngamma = self.ins.npix            # for slow-read coefs
                ncons = 2                         # num constraint equations
            else:
                ngamma = 0                        # no slow-read coefs
                ncons = 1                         # no constraints

            # total number of parameters and equations
            npars = nalpha+nbeta+ngamma  # number of unknowns
            neqns = npars+ncons          # total number of equations
            ndof = neqns                 # degrees of freedom
                        
            # set up linear problem
            b = np.zeros(neqns, dtype=float)
            A = np.zeros((neqns, npars), dtype=float)
            
            # master-master (term A)
            b[0] = np.nansum(wht*self.sci*self.sky)
            A[0,0] = np.nansum(wht*self.sky*self.sky)
                     
            # do the fast-read direction
            pix0 = self.ins.scipix[self.ins.fastaxis].start
            pix1 = self.ins.scipix[self.ins.fastaxis].stop
            
            k = np.arange(nalpha, nalpha+pix1-pix0, dtype=int)
            gammapix = nalpha+nbeta-self.ins.refpix
            for y, x, pix0, pix1 in self.ins.fast():
                # the vector term
                b[k] = np.nansum(wht[y, x]*self.sci[y, x], axis=self.ins.fastaxis)
                
                # cross term for fast-master (terms B and D)
                A[0, k] = A[k,0] = np.nansum(wht[y, x]*self.sky[y, x], axis=self.ins.fastaxis)
            
                # diagonal term for fast-fast (term E)
                A[k, k] = np.nansum(wht[y, x], axis=self.ins.fastaxis)
                               
                # update for slow-read if requested
                if fit_slow:
                    wf = wht[y, x]
                    kk = slice(gammapix+pix0, gammapix+pix1, 1)

                    # cross terms for fast-slow (terms H and F)            
                    if self.ins.fastaxis == 0:
                        A[k, kk] = wf.T
                        A[kk, k] = wf
                    else:
                        A[k, kk] = wf
                        A[kk, k] = wf.T

                # constraint terms for fast read
                A[npars, k] = float(pix1-pix0)

                # update index counter
                k += self.ins.npix
                
            # do the terms for slow-read if requested
            if fit_slow:
                k0 = nalpha+nbeta
                k = np.arange(k0, k0+self.ins.npix, dtype=int)
                
                # vector terms
                b[k] = np.nansum(wht[self.ins.scipix]*self.sci[self.ins.scipix], axis=self.ins.slowaxis)

                # cross term for master-slow (terms G/C)
                A[0, k] = A[k, 0] = np.nansum(wht[self.ins.scipix]*self.sky[self.ins.scipix], axis=self.ins.slowaxis)
                                                          
                # diagonal term for slow-slow (term K)
                A[k, k] = np.nansum(wht[self.ins.scipix], axis=self.ins.slowaxis)
                
                # constraint terms
                A[npars+1, k] = self.ins.npix
                
            # remove empty rows if they exist
            emptyrows = np.argwhere(np.all(A==0, axis=1)).flatten()
            if emptyrows.size > 0:
                b = np.delete(b, emptyrows, axis=0)
                A = np.delete(A, emptyrows, axis=0)
                A = np.delete(A, emptyrows, axis=1)
            # solve the sparse least-squares problem

            results = sparse.linalg.lsqr(A, b, damp=0.0)
        
            # solution vector
            x = results[0]
            istop = results[1]

            # if there were empty rows, set to default (0.0: primum non nocere)
            if emptyrows.size > 0:
                insidx = emptyrows-np.arange(emptyrows.size, dtype=int)
                x = np.insert(x, insidx, repval, axis=0)

            # get the solution vector
            alpha = x[0]
            beta = x[nalpha:nalpha+nbeta]
            gamma = x[nalpha+nbeta:nalpha+nbeta+ngamma]

            # the fast image
            fast = np.full_like(self.sci, np.nan, dtype=float)
            for k, (y, x, pix0, pix1) in enumerate(self.ins.fast()):
                k0 = k*self.ins.npix
                indices = slice(k0, k0+self.ins.npix)

                if self.ins.fastaxis == 0:
                    fast[y, x] = beta[indices]
                else:
                    fast[y, x] = beta[indices].reshape(self.ins.npix, 1)

            # the slow image
            slow = np.full_like(self.sci, np.nan, dtype=float)
            if fit_slow:
                slow[self.ins.scipix] = gamma
                gammaavg = np.average(slow[self.ins.scipix])
            else:
                slow[self.ins.scipix] = 0.0
                gammaavg = np.nan

            # do some quick checks
            betaavg = np.average(fast[self.ins.scipix])
            
        else:
            # doing the standard master sky
                
            num = np.nansum(wht[self.ins.scipix]*self.sci[self.ins.scipix]*self.sky[self.ins.scipix])
            den = np.nansum(wht[self.ins.scipix]*self.sky[self.ins.scipix]*self.sky[self.ins.scipix])
            alpha = num/den

            fast = np.full_like(self.sci, np.nan, dtype=float)
            fast[self.ins.scipix] = 0.0
            
            slow = np.full_like(self.sci, np.nan, dtype=float)
            slow[self.ins.scipix] = 0.0

            
            # set some dummy variables
            betaavg = np.nan
            gammaavg = np.nan
            ncons = 0                    # number of constraints
            npars = 1                    # number of unknowns
            neqns = npars+ncons
            istop = 0
            
        # master sky model
        mast = np.full_like(self.sci, np.nan, dtype=float)
        mast[self.ins.scipix] = alpha*self.sky[self.ins.scipix]
            
        # compute chi2
        model = mast+fast+slow
        resid = self.sci-model
        chi2 = np.nansum(wht[self.ins.scipix]*resid[self.ins.scipix]**2)
        nmeas = np.count_nonzero(wht[self.ins.scipix])
        ndof = nmeas-neqns
        redchi2 = chi2/ndof
  
        # some warning messages
        if alpha < self.ALPHATOL:
            warnings.warn("Master sky is negative.")
        if np.log10(max(np.abs(betaavg), 1e-8)) > self.BETATOL:
            warnings.warn(f"Fast 1/f is too large: (>{self.BETATOL}).")
        if np.log10(max(np.abs(gammaavg), 1e-8)) > self.GAMMATOL:
            warnings.warn(f"Slow 1/f is too large: (>{self.GAMMATOL}).")

        # save the results to a structure
        vers = im.version('WFSSBackground')
        self.results['bkgvers'] = (vers, 'Version of WFSSBackground')
        self.results['alpha'] = (alpha, 'master-sky coef')
        self.results['betaavg'] = (str(betaavg), 'ave of the fast read')
        self.results['gammaavg'] = (str(gammaavg), 'ave of the slow read')
        self.results['chi2'] = (chi2, 'chi2 of the fit')
        self.results['chi2nu'] = (redchi2, 'reduced chi2')
        self.results['npars'] = (npars, 'number of free parameters')
        self.results['neqns'] = (neqns, 'number of equations')
        self.results['ncons'] = (ncons, 'number of constraints')
        self.results['nmeas'] = (nmeas, 'number of measurements')
        self.results['ndof'] = (ndof, 'number of degrees of freedom')
        self.results['istop'] = (istop, 'see LSQR')        
        
        # print a useful message
        if verbose:
            self.print_results()
            
        if write:
            self.write(outfile=outfile)
        
        # how to return
        if combine:
            return model
        else:
            return mast, fast, slow

    def __getitem__(self, k):
        ''' just a shortcut for accessing results '''
        if k in self.results:
            return self.results[k][0]
        
    def write(self, model, outfile=None, **kwargs):
        # if a username is not passed, make one
        if not isinstance(outfile, str):
            # parse the file name
            pathname = os.path.dirname(self.scifile)
            basename = os.path.basename(self.scifile)
            basename = os.path.splitext(basename)[0]
            basename = '_'.join(basename.split('_')[:-1])

            # the output file name
            outfile = f'{basename}_{self.SUFFIX}.fits'
        
        try:
            # copy the file over
            shutil.copy(self.scifile, outfile)
            
            with fits.open(outfile, mode='update') as hdul:

                # update the primary header
                first = None
                for k, v in self.results.items():
                    hdul[0].header[k] = v
                    if not first:
                        first = k
                        
                kwargs['BITMASK'] = (self.bitmask, 'DQ value to mask')

                # update user supplied
                for k, v in kwargs.items():
                    hdul[0].header[k] = v

                # put header on this section
                hdul[0].header.set('', value='', before=first)
                hdul[0].header.set('', before=first,
                                   value='WFSS Background & 1/f Removal')
                hdul[0].header.set('', value='', before=first)

                # update the sci image by subtracting the model
                hdul[self.ins.sciext].data = self.sci-model
                
        except BaseException:
            warnings.warn(f'Cannot write file: {outfile}')
            outfile = None
            
        return outfile
   
        
    def print_results(self):
        print(f'     chi2 = {self.results["chi2"][0]}')
        print(f'  redchi2 = {self.results["chi2nu"][0]}')
        print(f'   master = {self.results["alpha"][0]}')    
        print(f'   <fast> = {self.results["betaavg"][0]}')     # should be ~0
        print(f'   <slow> = {self.results["gammaavg"][0]}')    # should be ~0
        print(f'     ndof = {self.results["ndof"][0]}')
        print('')

    def update_objmask(self, model, nsigma=3., min_size=16, outfile=None):
        
        res = (self.sci-model)/self.unc
        new = (res>=nsigma)

        new = morphology.remove_small_objects(new, min_size=min_size)

        # horizontal mask
        footprint = morphology.rectangle(*self.kern)
        new = morphology.binary_dilation(new, footprint=footprint)


        # update the mask
        self.obj = new | self.obj
        
        if isinstance(outfile, str):
            fits.writeto(outfile, self.obj.astype(int), overwrite=True)
            
        

