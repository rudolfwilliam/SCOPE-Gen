from scope_gen.scripts.format_data import format_data
from scope_gen.triviaqa.paths import DATA_DIR

def main(data_dir=DATA_DIR, output_dir="processed/data.pkl"):
    format_data(data_dir, output_dir)

if __name__ == "__main__":
    main()
