"""Inspect an explicit public catalog before building a spectral dataset.

The Ho et al. (2017) VizieR table contains LAMOST objects and transferred
labels. It is useful for label provenance and population selection, but it does
not provide paired APOGEE and LAMOST flux arrays. The real spectral benchmark
must still prepare LAMOST and DESI optical spectra under the NPZ contract in
``stellar_benchmark.data.cross_survey``.
"""

from stellar_benchmark.data.loaders import load_vizier_table


if __name__ == "__main__":
    catalog_id = "J/ApJ/836/5/table1"
    labels = load_vizier_table(catalog_id, max_rows=100)
    print(f"Loaded {len(labels)} rows from {labels.attrs['vizier_catalog_id']}")
    print("Columns:", list(labels.columns))
    print(labels.head())
    print(
        "\nThese are catalog labels, not a cross-survey spectral input. "
        "See configs/lamost_to_desi.yaml for the real optical-transfer contract."
    )
