from pcgen.data.format_data import format_data
from pcgen.mimic_cxr.paths import DATA_DIR

def main(data_dir=DATA_DIR, output_dir="processed/data.pkl"):
    format_data(data_dir=DATA_DIR, output_dir=output_dir, binarize_labels=True, label_threshold=0.4)

if __name__ == "__main__":
    main()
