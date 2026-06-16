import pandas as pd

file_path = r'd:\qqq\output\rebalance_day_2026-04-01_205633_strategy2\rebalance_day_report.xlsx'

# Get all sheet names
xl = pd.ExcelFile(file_path)
all_sheets = xl.sheet_names

print('='*80)
print('EXCEL FILE ANALYSIS')
print('='*80)
print(f'\nFile: {file_path}\n')
print('ALL SHEETS FOUND:')
for i, name in enumerate(all_sheets, 1):
    print(f'  {i}. {name}')

print('\n' + '='*80)
print('DETAILED SHEET ANALYSIS')
print('='*80)

for sheet_name in all_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f'\n--- Sheet: {sheet_name} ---')
    print(f'Rows: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    print('First 2 rows:')
    print(df.head(2).to_string())

print('\n' + '='*80)
print('CONFIRMATION CHECKS')
print('='*80)

# Check specific sheets exist
reb_day_exists = 'Rebalance_Day_Status' in all_sheets
reb_config_exists = 'Rebalance_Config_Status' in all_sheets
daily_returns_exists = 'Daily_Returns' in all_sheets
cumulative_returns_exists = 'Cumulative_Returns' in all_sheets

print(f'\n1. "Rebalance_Day_Status" exists: {reb_day_exists}')
print(f'   -> Expected: False (should NOT exist)')
print(f'   -> Result: {"PASS" if not reb_day_exists else "FAIL"}')

print(f'\n2. "Rebalance_Config_Status" exists: {reb_config_exists}')
print(f'   -> Expected: True (should exist)')
print(f'   -> Result: {"PASS" if reb_config_exists else "FAIL"}')

print(f'\n3. "Daily_Returns" exists: {daily_returns_exists}')
print(f'   -> Expected: True')
print(f'   -> Result: {"PASS" if daily_returns_exists else "FAIL"}')

print(f'\n4. "Cumulative_Returns" exists: {cumulative_returns_exists}')
print(f'   -> Expected: True')
print(f'   -> Result: {"PASS" if cumulative_returns_exists else "FAIL"}')

# Check Weight values in Current_Operations
if 'Current_Operations' in all_sheets:
    df_ops = pd.read_excel(file_path, sheet_name='Current_Operations')
    if 'Weight' in df_ops.columns:
        min_weight = df_ops['Weight'].min()
        all_above_threshold = (df_ops['Weight'] >= 0.0001).all()
        print(f'\n5. "Current_Operations" - All Weight >= 0.0001:')
        print(f'   Min Weight: {min_weight}')
        print(f'   All >= 0.0001: {all_above_threshold}')
        print(f'   -> Result: {"PASS" if all_above_threshold else "FAIL"}')
    else:
        print('\n5. "Current_Operations" - No "Weight" column found -> SKIP')
else:
    print('\n5. "Current_Operations" sheet not found -> SKIP')

# Check Weight values in All_Operations
if 'All_Operations' in all_sheets:
    df_all_ops = pd.read_excel(file_path, sheet_name='All_Operations')
    if 'Weight' in df_all_ops.columns:
        min_weight = df_all_ops['Weight'].min()
        all_above_threshold = (df_all_ops['Weight'] >= 0.0001).all()
        print(f'\n6. "All_Operations" - All Weight >= 0.0001:')
        print(f'   Min Weight: {min_weight}')
        print(f'   All >= 0.0001: {all_above_threshold}')
        print(f'   -> Result: {"PASS" if all_above_threshold else "FAIL"}')
    else:
        print('\n6. "All_Operations" - No "Weight" column found -> SKIP')
else:
    print('\n6. "All_Operations" sheet not found -> SKIP')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
