Finding Data
=====

There are several ways to create a ``FlareLightCurve`` in **AltaiPony**. All you need is your target's EPIC ID (and campaign). If you want to use de-trended K2SC light curves from archive or your computer, use ``from_K2SC_source`` or ``from_K2SC_file``. If you want to do the de-trending with **AltaiPony**, use ``from_TargetPixel_source``. You can also fetch ``KeplerLightCurve`` files. Their use in **AltaiPony** is limited for K2 light curves because sophisticated de-trending and quiescent flux modeling is needed (such as K2SC) to obtain reliable flare candidates:

.. module:: altaipony.lcio

.. automodapi:: altaipony.lcio
   :no-heading:
   :no-inherited-members:
   :no-inheritance-diagram:

 
   
