from .filter import Filter
import pandas as pd
import pickle

class PyTICS:
    def __init__(self, datafile, names_list,verbose=True, objname = 'MyAGN',AGN_ID = None):
        # Initialize a list of NestedObject instances with different names
        #self.filters = [FilterObject(name, default_value) for name in names_list]
        self.filters = {name: Filter(name) for name in names_list}
        self.verbose = verbose
        self.objname = objname
        self.AGN_ID = AGN_ID

        self.lco = pd.read_pickle(datafile)
        self.TEL = list(pd.unique(self.lco['telid']))

        self.Corrs = {} # initialise Corr output dict.
        # Optional parameters
        self.max_loops = 200
        self.frac = 0.5
        self.safe = 0.6
        self.bad_IDs = []
        self.star_lim = 100 
        
        if verbose: print('SCOPES:', self.TEL)
        print(" [PyTICS] Cleaning....")
        self.lco2 = self.Clean(self.lco)
    def __repr__(self):
        return f"PyTICS(nested_objects={self.filters})"
        

    def Calibrate(self):
        """ Do all the filters at once"""

        for name, filt in self.filters.items():
            if self.verbose: print(f"Processing {name}: {filt}")
            # Example operation: append '-processed' to the value of each object
            #filt.name += "-processed"
            DF, TR = filt.Corr(self.lco2, name, MAX_LOOPS = self.max_loops, bad_IDs = self.bad_IDs,
                    safe = self.safe, frac = self.frac, TEL = self.TEL, 
                        AGN_ID = self.AGN_ID, Star_Lim = self.star_lim)

            self.Corrs.update({"DF":DF, "TR":TR})



                       
    def Clean(self,Star_File):
        """ Drop duplicate values, existing zp and zp_err columns, and strange errors (= 99.0).
        
        Input:
        Star_File - Original data file (pd dataframe).
        
        Output:
        File2 - Cleaned up data file (pd dataframe).
        """
        
        File2 = Star_File.drop(columns = ["zp", "zp_err"])
        File2 = File2.drop_duplicates(subset=['id_apass', 'Filter', 'MJD', 
                                              'corr_aper', 'telid', 'airmass', 'seeing'], 
                                      keep='first', inplace=False, ignore_index=False)
        File2 = File2.drop(File2[(File2['err_aper'] > 20)].index)
        return(File2)
        
