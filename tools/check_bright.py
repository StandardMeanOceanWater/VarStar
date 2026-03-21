from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

Simbad.add_votable_fields('V', 'otype', 'sp')
c = SkyCoord(ra=128.0, dec=-47.9, unit='deg', frame='icrs')
r = Simbad.query_region(c, radius=3*u.deg)
if r:
    mask = np.array([v is not None and float(v) < 7 for v in r['V']])
    bright = r[mask]
    bright.sort('V')
    for row in bright:
        print(str(row['main_id']).ljust(25), f"V={float(row['V']):.2f}", str(row['otype']), str(row['sp_type']))
