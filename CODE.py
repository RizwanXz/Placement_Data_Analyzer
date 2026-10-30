import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================
# PROJECT SETTINGS
# =====================================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "placement.csv"

COLUMNS = ["Student", "Branch", "Company", "Package_LPA"]


# =====================================
# DATA CLEANING
# =====================================

def clean_data(raw_df):

    missing = [col for col in COLUMNS if col not in raw_df.columns]

    if missing:
        print("Missing columns:", ", ".join(missing))
        return pd.DataFrame(columns=COLUMNS), 0

    valid_rows = []
    skipped = 0
    seen = set()

    for row_no, row in enumerate(
        raw_df[COLUMNS].to_dict("records"), start=1
    ):

        student = str(row["Student"]).strip()
        branch = str(row["Branch"]).strip().upper()
        company = str(row["Company"]).strip()
        package_text = str(row["Package_LPA"]).strip()

        if not student or not branch or not company:
            print(f"Row {row_no} skipped: Empty field.")
            skipped += 1
            continue

        try:
            package = float(package_text)
        except (ValueError, TypeError):
            print(f"Row {row_no} skipped: Invalid package '{package_text}'.")
            skipped += 1
            continue

        if not pd.notna(package) or package <= 0:
            print(f"Row {row_no} skipped: Package must be positive.")
            skipped += 1
            continue

        key = (
            student.casefold(),
            branch,
            company.casefold(),
            package
        )

        if key in seen:
            print(f"Row {row_no} skipped: Duplicate record.")
            skipped += 1
            continue

        seen.add(key)

        valid_rows.append({
            "Student": student,
            "Branch": branch,
            "Company": company,
            "Package_LPA": package
        })

    result = pd.DataFrame(valid_rows, columns=COLUMNS)

    if not result.empty:
        result["Package_LPA"] = result["Package_LPA"].astype(float)

    return result, skipped


# =====================================
# CSV FILE LOADING
# =====================================

def load_data(file_path=CSV_FILE):

    try:
        raw_df = pd.read_csv(
            file_path,
            dtype=str,
            keep_default_na=False
        )

        return clean_data(raw_df)

    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")

    except pd.errors.EmptyDataError:
        print("Error: CSV file is empty.")

    except pd.errors.ParserError:
        print("Error: Invalid CSV format.")

    except OSError as error:
        print("File error:", error)

    return pd.DataFrame(columns=COLUMNS), 0


# =====================================
# PLACEMENT SUMMARY
# =====================================

def show_summary(df, skipped=0):

    if df.empty:
        print("No valid placement records.")
        print(f"Skipped records: {skipped}")
        return

    total = len(df)
    average = df["Package_LPA"].mean()
    highest = df.loc[df["Package_LPA"].idxmax()]

    print(f"Total placed: {total}")
    print(f"Average package: {average:.2f} LPA")

    print(
        f"Highest: {highest['Package_LPA']:.2f} LPA "
        f"({highest['Student']}, {highest['Company']})"
    )

    if skipped:
        print(f"Skipped invalid/duplicate rows: {skipped}")


# =====================================
# BRANCH ANALYSIS AND CHARTS
# =====================================

def branch_analysis(df, custom_limit=7.0, show_charts=True):

    if df.empty:
        print("No valid placement records.")
        return

    summary = df.groupby("Branch").agg(
        Placements=("Student", "count"),
        Average_Package=("Package_LPA", "mean")
    ).sort_index()

    placements = " | ".join(
        f"{branch}: {int(row['Placements'])}"
        for branch, row in summary.iterrows()
    )

    print(f"Placements: {placements}")

    averages = " | ".join(
        f"{branch} {row['Average_Package']:.2f}"
        for branch, row in summary.iterrows()
    )

    print(f"Average package: {averages}")

    above = df[df["Package_LPA"] > custom_limit]

    if above.empty:
        print(f"Above {custom_limit:g} LPA: No students found.")
    else:
        names = " | ".join(
            f"{row['Student']} ({row['Package_LPA']:.2f})"
            for _, row in above.iterrows()
        )

        print(f"Above {custom_limit:g} LPA: {names}")

    # BAR CHART

    plt.figure(figsize=(7, 4))

    bars = plt.bar(
        summary.index,
        summary["Placements"]
    )

    plt.title("Branch-wise Placements")
    plt.xlabel("Branch")
    plt.ylabel("Number of Placements")
    plt.bar_label(bars)
    plt.tight_layout()

    plt.savefig(BASE_DIR / "branch_placements.png")

    # PIE CHART

    plt.figure(figsize=(6, 6))

    plt.pie(
        summary["Placements"],
        labels=summary.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Branch-wise Placements")
    plt.tight_layout()

    plt.savefig(BASE_DIR / "branch_placements_pie.png")

    print("Charts saved successfully.")

    if show_charts:
        plt.show()

    plt.close("all")


# =====================================
# TEST CASE 1: NORMAL INPUT
# =====================================

def run_tc1():

    print("\n--- TC1: Normal Input ---")

    raw = pd.DataFrame([
        ["Aman", "CSE", "TCS", "6"],
        ["Riya", "CSE", "Infosys", "8"],
        ["Karan", "ECE", "Wipro", "4"]
    ], columns=COLUMNS)

    df, skipped = clean_data(raw)
    show_summary(df, skipped)


# =====================================
# TEST CASE 2: MINIMUM INPUT
# =====================================

def run_tc2():

    print("\n--- TC2: Minimum Input ---")

    raw = pd.DataFrame([
        ["Neha", "CSE", "TCS", "3.5"]
    ], columns=COLUMNS)

    df, skipped = clean_data(raw)
    show_summary(df, skipped)


# =====================================
# TEST CASE 3: INVALID INPUT
# =====================================

def run_tc3():

    print("\n--- TC3: Invalid Input ---")

    raw = pd.DataFrame([
        ["Aman", "CSE", "TCS", "abc"],
        ["Riya", "CSE", "Infosys", "8"],
        ["Karan", "ECE", "Wipro", "-4"]
    ], columns=COLUMNS)

    df, skipped = clean_data(raw)

    total = len(df)

    print(f"Valid record: {total} | Total placed: {total}")
    print(f"Average package: {df['Package_LPA'].mean():.2f} LPA")


# =====================================
# TEST CASE 4: BRANCH ANALYSIS
# =====================================

def run_tc4():

    print("\n--- TC4: Branch Analysis ---")

    raw = pd.DataFrame([
        ["Aman", "CSE", "TCS", "6"],
        ["Riya", "CSE", "Infosys", "8"],
        ["Karan", "ECE", "Wipro", "4"],
        ["Ravi", "ECE", "Accenture", "12"]
    ], columns=COLUMNS)

    df, skipped = clean_data(raw)

    print(f"Valid records: {len(df)}")
    print(f"Skipped records: {skipped}")

    branch_analysis(
        df,
        custom_limit=7,
        show_charts=True
    )


# =====================================
# MAIN MENU
# =====================================

def main():

    print("\n====================================")
    print("       PLACEMENT DATA ANALYZER")
    print("====================================")

    while True:

        print("\n========== TEST CASE MENU ==========")
        print("1. Run TC1 - Normal Input")
        print("2. Run TC2 - Minimum Input")
        print("3. Run TC3 - Invalid Input")
        print("4. Run TC4 - Branch Analysis")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            run_tc1()

        elif choice == "2":
            run_tc2()

        elif choice == "3":
            run_tc3()

        elif choice == "4":
            run_tc4()

        elif choice == "5":
            print("Thank you for using Placement Data Analyzer!")
            break

        else:
            print("Invalid choice. Enter 1 to 5.")


# =====================================
# PROGRAM START
# =====================================

if __name__ == "__main__":
    main()