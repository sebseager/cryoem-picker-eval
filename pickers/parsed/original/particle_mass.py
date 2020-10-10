import os
import sys
import glob
import numpy as np

from optparse import OptionParser, SUPPRESS_HELP

def ext_param(filename , header=6 , loc=4):
    with open(filename , 'r') as f:
        lines = f.readlines()
    lst = []
    for line in lines[header:]:
        line_t = line.split()
        lst.append(line_t[loc])
    return lst

def sel_param(filenameC , header=6 , loc=4 , cutoff = 0. , newsuffix = 'null', oldsuffix='null'):
    with open(filenameC + '%s_param.star' %oldsuffix, 'r') as f:
        lines = f.readlines()
    lst = []
    if newsuffix != 'null':
        with open(filenameC + '%s_param.star' %newsuffix, 'w') as f:
            with open(filenameC + '%s.star' %newsuffix, 'w') as fo:
                for line in lines[:header]:
                    f.write(line)
                    fo.write(line)
                for line in lines[header:]:
                    line_t = line.split()
                    if float(line_t[loc]) >= cutoff:
                        f.write(line)
                        fo.write('\t'.join([line_t[0],line_t[1]]) + '\n')
    return 0

if __name__ == '__main__':
    progName = os.path.basename(sys.argv[0])
#    usage = progName + """ <train|valid|pick> [options]
    usage = progName + """ <drawmass|cutoff> [options]
    This program generates mass distribution of particles picked by parsed_main.py.

    PARSED is a deep-learning model that reads a list of MRC files of cryo-EM micrographs, 
    and then automatically picks particles of biological macromolecules in these micrographs. 
    The picked particles could be directly imported into 3D reconstruction programs such as 
    cryoSPARC or RELION. 

    Reference:
    R. Yao, J. Qian, Q. Huang. A universal deep-learning model for automated
    cryo-EM particle picking (To be published).
    """
    
    parser = OptionParser(usage , version='PARSED 0.0.1')
    
    #Job Parameters
    parser.add_option("--pick_output", metavar='PATH_OUTPUT', default=None, type='string', help="folder path for PARSED results")
    parser.add_option("--job_suffix", metavar='demo0', default='demo0', type='string', help = "file suffix in PARSED results")
    parser.add_option("--output_suffix", metavar="PATH_OUTPUT", default='checked', type='string', help="checked coordinates file suffix (default checked)")
    parser.add_option("--thres", metavar=0, default=0, type=np.int, help = "threshold on mass (default 0)")
    parser.add_option("--tmp_hist",metavar='tmp_hist.jpg',default='tmp_hist.jpg',type='string', help="temporary file for mass distribution graph")
    parser.add_option("--tmp_mass",metavar='tmp_mass.lst',default='tmp_mass.lst',type='string', help="temporary file for mass")

    (global_options, global_args) = parser.parse_args()
    command = ' '.join(sys.argv)
    
    global_mode = sys.argv[1]
    
    path = global_options.pick_output
    nameList = glob.glob(path + '/*%s_param.star' %global_options.job_suffix)

    header = 6
    # +2 x y mass size ecc signal raw_mass ep prob mask
    loc = 4 #mass
    #loc = 5 #size
    
    # non X11
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    import multiprocessing
    
    
    extPool = multiprocessing.Pool()
    extRes = []
    for name_i in nameList:
        extRess = extPool.apply_async(ext_param,args=(name_i,header,loc))
        extRes.append(extRess)

    extPool.close()
    extPool.join()

    extResS = []
    for extResss in extRes:
        extRessS = extResss.get()
        extResS.append(extRessS)

    extResF = []
    for extRessS in extResS:
        extResF += extRessS
    extResF = list(map(lambda x: float(x), extResF))
    
    if global_mode == 'drawmass':
        fig, ax = plt.subplots(frameon=True, figsize=(6 , 5))
        #plt.hist(extResF , bins = np.linspace(0,150,400))
        hist = ax.hist(extResF , bins = range(0,150,1), weights=np.zeros_like(extResF) + 1. / len(extResF) )
        ax.xaxis.set_major_locator( MultipleLocator(10) )
        ax.xaxis.set_minor_locator( MultipleLocator(2) )
        ax.yaxis.set_major_locator( MultipleLocator(0.01) )
        ax.yaxis.set_minor_locator( MultipleLocator(0.002) )
        
        ax.axis([-0, 150,0,0.06])
        ax.grid(color='gray' , linestyle='dotted', linewidth= 0.5 , alpha = 0.5)
        ax.xaxis.set_tick_params(direction='in');ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_tick_params(direction='in');ax.yaxis.set_ticks_position('left')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        fig.savefig('%s' %global_options.tmp_hist)
        
        with open('%s' %global_options.tmp_mass, 'w') as f:
            for item in extResF:
                f.write(str(item)+'\n')

    elif global_mode == 'cutoff':
        cutoff = global_options.thres
        len_suffix = len('%s_param.star' %global_options.job_suffix)
        nameListC = list(map(lambda x: x[:-len_suffix] , nameList))
        
        selPool = multiprocessing.Pool()
        selRes = []
        for nameListC_i in nameListC:
            selRess = selPool.apply_async(sel_param,args=(nameListC_i,header,loc,cutoff,global_options.output_suffix,global_options.job_suffix))
            selRes.append(selRess)

        selPool.close()
        selPool.join()

        selResS = []
        for selResss in selRes:
            selRessS = selResss.get()
            selResS.append(selRessS)
                
        with open('%s' %global_options.tmp_mass, 'w') as f:
            for item in extResF:
                f.write(str(item)+'\n')

    else:
        print("Unrecognized mode choice !!")
