from EDA import eda, eda_plots, eda_analysis, explore
from SRC import evaluate
import uvicorn


def main():
    print("=== Step 1: EDA ===")
    explore.run()
    eda.run()
    eda_plots.run()
    eda_analysis.run()

    print("\n=== Step 2: Preprocessing + Training + Evaluation ===")
    evaluate.run()

    print("\n=== Pipeline Complete ✓ ===")

    print("\n=== Step 3: Starting API ===")
    print("Open: http://127.0.0.1:8000/docs")

    from SRC.api import app
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()