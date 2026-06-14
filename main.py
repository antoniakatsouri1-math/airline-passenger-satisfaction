from EDA import eda, eda_plots, eda_analysis, explore
from SRC import evaluate

def main():
    print("=== Step 1: EDA ===")
    explore.run()
    eda.run()
    eda_plots.run()
    eda_analysis.run()

    print("\n=== Step 2: Preprocessing + Training + Evaluation ===")
    evaluate.run()

    print("\n=== Pipeline Complete ✓ ===")

if __name__ == '__main__':
    main()