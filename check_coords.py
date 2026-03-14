from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

Simbad.add_votable_fields('V', 'otype')
targets = {
    'AlVel_folder': (128.005, -47.929),
    'CCAnd_folder': ( 10.760,  41.350),
}

for name, (ra, dec) in targets.items():
    c = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    result = Simbad.query_region(c, radius=6*u.arcmin)
    if result:
        result.sort('V')
        print(f'\n{name} ({ra:.3f}, {dec:.3f}):')
        for row in result[:5]:
            print(f'  {str(row["main_id"]):30s}  V={row["V"]}  type={row["otype"]}')
    else:
        print(f'\n{name}: no result')
