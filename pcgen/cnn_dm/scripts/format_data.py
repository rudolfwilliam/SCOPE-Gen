from pcgen.scripts.format_data import format_data
from pcgen.cnn_dm.paths import DATA_DIR

def main(data_dir=DATA_DIR, output_dir="processed/data.pkl"):
    format_data(data_dir, output_dir, binarize_labels=True, label_threshold=0.35)

if __name__ == "__main__":
    main()
    