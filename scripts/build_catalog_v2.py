"""Build structured Egyptian medicine catalog: name, strength(s), form, pack, ingredients."""
import csv, re, json, sys, sqlite3, collections

SRC = sys.argv[1]
UNIT = r'(MCG|MG|GM|G|KG|ML|L|IU|I\.U\.?|UNITS?|U|%|MEQ|MMOL|MIU)'
NUM = r'(\d+(?:[.,]\d+)?)'
# 500 MG | 250MG/5ML | 5 MG/ML | 0.1 % | 1 GM | 400/80 MG | 20 MG/12.5 MG
RATIO = re.compile(rf'{NUM}\s*{UNIT}\s*/\s*{NUM}?\s*{UNIT}', re.I)
SINGLE = re.compile(rf'{NUM}\s*{UNIT}\b', re.I)
COMBO = re.compile(rf'{NUM}(?:\s*/\s*{NUM})+\s*{UNIT}', re.I)
PACK = re.compile(r'\b(\d+)\s*(F\.?C\.?\s*TABS?|TABS?|TABLETS?|CAPS?|CAPSULES?|AMPS?|AMPOULES?|VIALS?|SACHETS?|SUPP|SUPPOSITORIES|LOZ|LOZENGES?|PATCHES|PATCH|STRIPS?|PREFILLED\s*SYRINGES?|PEN)\b', re.I)
FORMS = [
 ('tablet', r'TAB|F\.?C\.?T|CHEW'), ('capsule', r'CAP'), ('syrup', r'SYRUP|SYR\b'),
 ('suspension', r'SUSP'), ('drops', r'DROPS?|ORAL DROP'), ('injection', r'AMP|VIAL|INJ|I\.?V|I\.?M|PFS|SYRINGE|PEN'),
 ('cream', r'CREAM'), ('ointment', r'OINT'), ('gel', r'GEL'), ('suppository', r'SUPP'),
 ('sachet', r'SACHET'), ('spray', r'SPRAY'), ('eye', r'EYE|OPHTH'), ('lotion', r'LOTION'),
 ('shampoo', r'SHAMPOO'), ('inhaler', r'INHAL|MDI|NEBUL'), ('patch', r'PATCH'), ('solution', r'SOLN|SOLUTION'),
 ('lozenge', r'LOZ'),
]

def norm_unit(u):
    u = u.upper().replace('.', '')
    return {'GM':'g','G':'g','MG':'mg','MCG':'mcg','ML':'ml','L':'l','IU':'IU','I U':'IU','U':'IU','UNIT':'IU','UNITS':'IU','%':'%','MEQ':'mEq','MMOL':'mmol','MIU':'MIU'}.get(u,u.lower())

def parse_strengths(text):
    """Return list of strength strings like '500 mg', '250 mg/5 ml', '400/80 mg'."""
    out, t = [], text
    for m in RATIO.finditer(t):
        n1,u1,n2,u2 = m.group(1),m.group(2),m.group(3),m.group(4)
        # skip pack/volume like '10 MG/ML' is valid; '120 ML' alone is volume handled below
        out.append(f"{n1} {norm_unit(u1)}/{(n2+' ') if n2 else ''}{norm_unit(u2)}")
    t2 = RATIO.sub(' ', t)
    for m in COMBO.finditer(t2):
        nums = re.findall(r'\d+(?:[.,]\d+)?', m.group(0))
        out.append('/'.join(nums)+' '+norm_unit(m.group(m.lastindex if False else 0).split()[-1] if False else re.search(UNIT+r'$', m.group(0), re.I).group(1)))
    t3 = COMBO.sub(' ', t2)
    vols = []
    for m in SINGLE.finditer(t3):
        s = f"{m.group(1)} {norm_unit(m.group(2))}"
        (vols if norm_unit(m.group(2)) in ('ml','l') else out).append(s)
    return out, vols

def parse_form(name, route):
    for f,pat in FORMS:
        if re.search(r'\b(?:'+pat+r')', name, re.I): return f
    r = (route or '').upper()
    if 'SOLID' in r: return 'solid'
    if 'LIQUID' in r: return 'liquid'
    return (route or '').lower()

def split_ingredients(sci):
    sci = re.sub(r'\s+', ' ', sci or '').strip()
    parts = [p.strip() for p in sci.split('+') if p.strip()]
    ings = []
    for p in parts:
        # ingredient may carry its own strength: "VITAMIN C 1 GM"
        s,_ = parse_strengths(p)
        base = SINGLE.sub('', RATIO.sub('', p)).strip(' ,-')
        ings.append({'name': base.title(), 'strength': s[0] if s else None})
    return ings

rows, stats = [], collections.Counter()
with open(SRC, newline='', encoding='utf-8-sig') as f:
    for i, r in enumerate(csv.DictReader(f), 1):
        name = (r['commercial_name_en'] or '').strip()
        if not name: stats['no_name'] += 1; continue
        strengths, vols = parse_strengths(name)
        ings = split_ingredients(r['scientific_name'])
        if not strengths:
            strengths = [x['strength'] for x in ings if x['strength']]
        pack = PACK.search(name)
        # brand = text before first number
        brand = re.split(r'\s\d', name, 1)[0].strip().title()
        rec = {
          'id': i, 'brand': brand, 'name_en': name, 'name_ar': (r['commercial_name_ar'] or '').strip(),
          'ingredients': ings, 'strengths': strengths, 'volume': vols[0] if vols else None,
          'form': parse_form(name, r['route']), 'pack': (pack.group(1) if pack else None),
          'route': r['route'], 'drug_class': (r['drug_class'] or '').strip(),
          'manufacturer': (r['manufacturer'] or '').strip(),
          'price_egp': float(r['price_egp']) if r['price_egp'] not in ('', None) else None,
        }
        stats['rows'] += 1
        stats['with_strength'] += bool(strengths)
        stats['with_volume'] += bool(vols)
        rows.append(rec)

json.dump(rows, open('/home/claude/out/medicines.json','w',encoding='utf-8'), ensure_ascii=False)

# SQLite for the Android app (Room/SQLite): fast search by brand, Arabic name, ingredient
db = '/home/claude/out/medicines.db'
import os
if os.path.exists(db): os.remove(db)
c = sqlite3.connect(db)
c.executescript('''
CREATE TABLE medicine(id INTEGER PRIMARY KEY, brand TEXT, name_en TEXT, name_ar TEXT, ingredients TEXT,
  strengths TEXT, volume TEXT, form TEXT, pack TEXT, route TEXT, drug_class TEXT, manufacturer TEXT, price_egp REAL);
CREATE INDEX idx_brand ON medicine(brand);
CREATE INDEX idx_ing ON medicine(ingredients);
''')
for r in rows:
    c.execute('INSERT INTO medicine VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', (
      r['id'], r['brand'], r['name_en'], r['name_ar'],
      '; '.join(i['name'] for i in r['ingredients']), '; '.join(r['strengths']), r['volume'],
      r['form'], r['pack'], r['route'], r['drug_class'], r['manufacturer'], r['price_egp']))
c.commit(); c.close()

# distinct active ingredients (for DDI matching) with all strengths seen
ing = collections.defaultdict(set)
for r in rows:
    if len(r['ingredients']) == 1:
        for s in r['strengths']: ing[r['ingredients'][0]['name']].add(s)
    else:
        for x in r['ingredients']: ing[x['name']]
json.dump({k: sorted(v) for k,v in ing.items()}, open('/home/claude/out/ingredients.json','w'), ensure_ascii=False)
print(dict(stats), 'distinct ingredients:', len(ing))
