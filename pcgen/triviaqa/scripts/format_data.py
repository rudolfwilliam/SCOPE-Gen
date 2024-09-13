from pcgen.data import format_data
from pcgen.triviaqa.paths import DATA_DIR

def main(data_dir=DATA_DIR, output_dir="processed/data.pkl"):
    format_data(data_dir, output_dir)

if __name__ == "__main__":
    main()
