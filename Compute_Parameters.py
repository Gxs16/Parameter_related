#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ef5_kw_estimate import ReadGrid, WriteGrid
import numpy as np
import pickle


# # Compute alpha0
# ## Requirements
# * slope   *'KW_parameters/DEM/slope_3rds.tif'*
# * manning   *'KW_parameters/data/manningn.tif'*
# * COEM
# 
# ## Formulas
# $$
# COEM = \frac{1}{maning}\\
# alpha0 = (\frac{1}{COEM\times\sqrt{\frac{slope}{100}}})^{\frac{3}{5}}
# $$

# In[ ]:


slope = ReadGrid('KW_parameters/DEM/slope_3rds.tif', keepInfo=True)
maning = ReadGrid('KW_parameters/data/manningn.tif')


# In[ ]:


Coem = 1/maning
alpha0 = (1/Coem/(slope/100)**0.5)**(3/5)
WriteGrid('KW_parameters/data/alpha0.tif', alpha0)


# # Compute under and leaki
# ## Requirements
# * KSAT *'CREST_Parameters/ksat.tif'*
# * leaki = 0.003
# 
# ## Formulas
# $$
# under = KSAT \times 0.001\\
# leaki = 0.003
# $$

# In[ ]:


KSAT = ReadGrid('CREST_Parameters/ksat.tif',keepInfo=True)
under = KSAT * 0.001
leaki = KSAT
leaki[:] = 0.003

# In[ ]:


WriteGrid('CREST_Parameters/under.tif', under)
WriteGrid('CREST_Parameters/leaki.tif', leaki)'''


# # Compute alpha and beta
# ## Requirements
# 
# This module uses information from the USGS stations over the CONUS to generate a statistical model for estimating the kinematic wave alpha and beta parameters for a given grid. The already trained model is provided here.
# 
# This model requires basin averaged estimates of the following variables:
# * Mean annual temperature (degrees C) *EF5-KW-Estimation-master/input_grids/temp.avg.tif*
# * Mean annual precipitation (mm)*EF5-KW-Estimation-master/input_grids/precip.avg.tif*
# * Impervious area (%) *EF5-KW-Estimation-master/input_grids/imperv.avg.tif*
# * Clay (%) *EF5-KW-Estimation-master/input_grids/clay_pct.avg.tif*
# * Sand (%) *EF5-KW-Estimation-master/input_grids/sand_pct.avg.tif*
# * Silt (%) *EF5-KW-Estimation-master/input_grids/silt_pct.avg.tif*
# * Relief ratio *EF5-KW-Estimation-master/input_grids/relief.ratio.avg.tif*
# * Course Fragments (%) *EF5-KW-Estimation-master/input_grids/course_fragments.avg.tif*
# * Depth to bedrock (mm) *EF5-KW-Estimation-master/input_grids/abs_depth_bedrock.avg.tif*
# * Bulk Density *EF5-KW-Estimation-master/input_grids/bulk_density.avg.tif*
# * Population *EF5-KW-Estimation-master/input_grids/population.avg.tif*

# In[ ]:


# keepInfo=True keeps the projection and spatial reference information for this grid...
# to use when writing the output grids
# We need log basin area, so compute that here
basinArea = ReadGrid("EF5-KW-Estimation-master/input_grids/basin.area.tif", keepInfo=True)
basinArea = np.log10(basinArea)

temp = ReadGrid("EF5-KW-Estimation-master/input_grids/temp.avg.tif")
precip = ReadGrid("EF5-KW-Estimation-master/input_grids/precip.avg.tif")
imperv = ReadGrid("EF5-KW-Estimation-master/input_grids/imperv.avg.tif")
clay = ReadGrid("EF5-KW-Estimation-master/input_grids/clay_pct.avg.tif")
sand = ReadGrid("EF5-KW-Estimation-master/input_grids/sand_pct.avg.tif")
silt = ReadGrid("EF5-KW-Estimation-master/input_grids/silt_pct.avg.tif")
rr = ReadGrid("EF5-KW-Estimation-master/input_grids/relief.ratio.avg.tif")
frags = ReadGrid("EF5-KW-Estimation-master/input_grids/course_fragments.avg.tif")
bedrock = ReadGrid("EF5-KW-Estimation-master/input_grids/abs_depth_bedrock.avg.tif")
density = ReadGrid("EF5-KW-Estimation-master/input_grids/bulk_density.avg.tif")
pop = ReadGrid("EF5-KW-Estimation-master/input_grids/population.avg.tif")


# ## Load the pickled models so we can run them on our data

# In[ ]:


scaler = pickle.load(open("scaler.p", "rb"))
alphamod = pickle.load(open("alpha_model.p", "rb"))
betamod = pickle.load(open("beta_model.p", "rb"))


# ## Run the models to estimate alpha & beta

# In[ ]:


print("Transforming to scaled parameter space")
pred_real = scaler.transform(np.column_stack((basinArea, temp, precip, clay, sand, silt, rr, frags, bedrock, imperv, density, pop)))
del basinArea
del temp
del precip
del clay
del sand
del silt
del rr
del frags
del bedrock
del imperv
del density
del pop

print("Computing alpha")
alpha = alphamod.predict(pred_real)
alpha = np.power(10.0, alpha)

print("Computing beta")
beta = betamod.predict(pred_real)

# Since the model extrapolates, we do a trick here to bound the minimum beta to 0.
bad = np.where(beta < 0.0)
beta[bad] = np.exp(beta[bad] * 10.0)


# ## Write the new parameter grids to disk

# In[ ]:


WriteGrid("EF5-KW-Estimation-master/output_grids/kw_alpha.tif", alpha)
WriteGrid("EF5-KW-Estimation-master/output_grids/kw_beta.tif", beta)

