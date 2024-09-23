from pcgen.data.format_data import labels_to_losses, format_data
from pcgen.molecules.paths import DATA_DIR

def main(data_dir=DATA_DIR, output_dir="processed/data.pkl"):
    labels_to_losses(data_dir)
    format_data(data_dir, output_dir=output_dir)

if __name__ == "__main__":
    main(output_dir="processed/data.pkl")
