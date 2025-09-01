#!/usr/bin/env python3
"""
IceCube Data Analysis - Quick Start Example
This script demonstrates how to use the ParquetExplorer to analyze your IceCube data.
"""

from initial_data_exploration.parquet_reader import ParquetExplorer, explore_all_parquet_files

def main():
    print("🧊 IceCube Data Analysis - Quick Start")
    print("=" * 50)
    
    # First, let's explore all parquet files in the current directory
    explorers = explore_all_parquet_files(directory='data')
    
    if not explorers:
        print("No parquet files found. Please make sure you have .parquet files in this directory.")
        return
    
    # Let's dive deeper into each file
    for filename, explorer in explorers.items():
        print(f"\n🔍 Deep dive into: {filename}")
        print("=" * 60)
        
        # Get basic information
        basic_info = explorer.get_basic_info()
        print("\n📊 Basic File Information:")
        for key, value in basic_info.items():
            print(f"  {key}: {value}")
        
        # Get schema information  
        schema_info = explorer.get_schema_info()
        print(f"\n📋 Schema ({schema_info['total_columns']} columns):")
        for col in schema_info['columns'][:15]:  # Show first 15 columns
            print(f"  • {col['name']:<20} | {col['type']:<15} | Nullable: {col['nullable']}")
        
        if len(schema_info['columns']) > 15:
            print(f"  ... and {len(schema_info['columns']) - 15} more columns")
        
        # Preview data (load a small sample)
        print(f"\n📋 Data Preview:")
        explorer.preview_data(nrows=5, library="pandas")
        
        # Get data summary
        print(f"\n📈 Data Summary (first 1000 rows):")
        summary = explorer.get_data_summary("pandas", 1000)
        print(f"  Shape: {summary['shape']}")
        print(f"  Memory usage: {summary['memory_usage_mb']} MB")
        print(f"  Data types: {len(set(summary['dtypes'].values()))} unique types")
        
        # Show null counts for columns with missing data
        null_counts = {k: v for k, v in summary['null_counts'].items() if v > 0}
        if null_counts:
            print(f"\n❗ Columns with missing data:")
            for col, count in list(null_counts.items())[:10]:  # Show first 10
                print(f"  • {col}: {count} nulls")
        else:
            print(f"\n✅ No missing data found!")
        
        # Export detailed report
        print(f"\n📄 Exporting detailed report...")
        report_path = explorer.export_summary_report(f"{filename}_analysis_report.json")
        
        print(f"\n" + "="*60)

if __name__ == "__main__":
    main()
