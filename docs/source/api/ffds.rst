Flare Frequency Distributions and Power Laws
=====

Once you have found all the flares, you can compute statistical measures using your flare table. You can directly use the `FlareLightCurve.flares` table, or any `pandas` table where the recovered flare energies column is named `ed_rec`.

Assume we have such a `FlareLightCurve` called `flc` with the required attribute `flares`, we can create a FFD object 

::

    from altaipony.ffd import FFD
    simple_ffd = FFD(f=flc.flares)

`simple_ffd.f` contains the table with the energies of the flares. We can also specify the observing time it took to detect the event listed in the table:

::

    simple_ffd.tot_obs_time = 20.
    
The unit is your choice, and you should know which one you are using. If you do not specify `tot_obs_time`, the FFD frequencies will instead be the number counts, i.e. `simple_ffd.tot_obs_time=1.`.
