from scope_gen.scripts.format_data import format_data
from scope_gen.mimic_cxr.paths import DATA_DIR

def main(data_dir=DATA_DIR, output_dir="processed/data.pkl"):
    format_data(data_dir=DATA_DIR, output_dir=output_dir, binarize_labels=True, label_threshold=0.6)

if __name__ == "__main__":
    main()
