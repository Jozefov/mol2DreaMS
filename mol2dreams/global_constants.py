
# Continuous Feature Ranges
PRECURSOR_MZ_MAX = 1000.0

# Binned Feature Ranges
COLLISION_ENERGY_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 10 bins: 0-10, 10-20, ..., 90-100

# Categorical Feature Values
ADDUCT_CATEGORIES = ['[M+H]+', '[M+Na]+']
INSTRUMENT_TYPE_CATEGORIES = ['Orbitrap', 'QTOF']

# Number of Categories (including 'unknown')
ADDUCT_NUM_CLASSES = len(ADDUCT_CATEGORIES) + 1  # +1 for 'unknown'
INSTRUMENT_TYPE_NUM_CLASSES = len(INSTRUMENT_TYPE_CATEGORIES) + 1  # +1 for 'unknown'

# Binned COLLISION_ENERGY Encoding
COLLISION_ENERGY_NUM_BINS = len(COLLISION_ENERGY_BINS) - 1  # 10 bins