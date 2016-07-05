import os
import subprocess
import pandas as pd
import numpy as np
import re
from os.path import join as opj

'''
list of .fam for eur set that have no freesurfer data
or where it seems like there is duplicate data:

------------------------------------------------------
POSSIBLE DUPLICATES:

FOUND IN: scz_umcu_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
   * scz_paco_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
   
------------------------------------------------------
 
CANNOT FIND IDS IN FS FILE:
   * scz_cati_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
   * scz_mfs1_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
   * scz_mcic_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
   * scz_page_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
   * scz_cogs_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
------------------------------------------------------

EXCEPTIONS:

   * scz_mts1_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
       only 2 participants that match the freesurfer csv
       
   * scz_gapl_eur_gb-qc.hg19.ch.fl.bgn.reid.fam
       can only find 1, and it seems to be a 
       duplicate ID from another file
------------------------------------------------------
''';

#fscsv = '/data/petryshen/genus/freesurfer/GENUS_all_FreeSurfer_phenotypes.csv'
#bdgen = '/data/petryshen/genus/gwas/imputed/{}/cobg_dir_genome_wide/'
fscsv = '/om/user/ysa/genus_dev/GENUS_all_FreeSurfer_phenotypes.csv'
bdgen = '/om/user/ysa/genus_dev/{}/cobg_dir_genome_wide/'
eurgen = bdgen.format('genus1_imp_eur_1000GP_P1_allchr')
allgen = bdgen.format('genus1_imp_all_1000GP_P1_allchr')

# 1149 participants
data_exist_fs = [
    'scz_camh_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_umcu_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_nefs_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_deco_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_cida_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_lanr_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_phrs_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
    'scz_tcdn_eur_gb-qc.hg19.ch.fl.bgn.reid.fam',
]

def pheno_plus_geno(genfile, fscsv, case=None):
    gen = pd.read_csv(genfile, sep=' ', header=None)
    gen.set_index(gen.iloc[:,1], inplace=True)
    gen.columns = ['FID','IID','PFa','PMo','sex','affected']
    fsdf = pd.read_csv(fscsv, low_memory=False)
    fsdf.set_index(fsdf['IID'], inplace=True)
    return pd.merge(gen, fsdf, on='IID')

pg_list = []

for i in data_exist_fs:
    pg_list.append(pheno_plus_geno(opj(eurgen,i), fscsv))

fs_fam = pd.concat([pg_list[0], pg_list[1],
                      pg_list[2], pg_list[3],
                      pg_list[4], pg_list[5],
                      pg_list[6], pg_list[7]], 
                   ignore_index=True)

