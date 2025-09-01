import pyarrow.parquet as pq
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Any, Optional
import json


class ParquetExplorer:
    """A comprehensive parquet file explorer for IceCube data analysis."""
    
    def __init__(self, file_path: str):
        """Initialize the explorer with a parquet file path."""
        self.file_path = Path(file_path)
        self.file_info = None
        self.pandas_df = None
        self.polars_df = None
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load basic file information
        self._load_file_info()
    
    def _load_file_info(self):
        """Load basic file information using PyArrow."""
        try:
            self.file_info = pq.read_metadata(self.file_path)
            print(f"✅ Successfully loaded file: {self.file_path.name}")
        except Exception as e:
            print(f"❌ Error loading file info: {e}")
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic information about the parquet file."""
        if not self.file_info:
            return {}
        
        info = {
            "file_name": self.file_path.name,
            "file_size_mb": round(self.file_path.stat().st_size / (1024 * 1024), 2),
            "num_rows": self.file_info.num_rows,
            "num_columns": self.file_info.num_columns,
            "num_row_groups": self.file_info.num_row_groups,
            "format_version": self.file_info.format_version,
            "created_by": self.file_info.created_by,
        }
        
        return info
    
    def get_schema_info(self) -> Dict[str, Any]:
        """Get detailed schema information."""
        if not self.file_info:
            return {}
        
        schema = self.file_info.schema.to_arrow_schema()
        columns = []
        
        for i, field in enumerate(schema):
            columns.append({
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
                "index": i
            })
        
        return {
            "total_columns": len(columns),
            "columns": columns
        }
    
    def load_pandas(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load data as pandas DataFrame with optional row limit."""
        try:
            if nrows:
                # Read only first n rows for quick exploration
                table = pq.read_table(self.file_path, use_threads=True)
                self.pandas_df = table.slice(0, nrows).to_pandas()
                print(f"📊 Loaded {len(self.pandas_df)} rows (limited) into pandas DataFrame")
            else:
                self.pandas_df = pd.read_parquet(self.file_path)
                print(f"📊 Loaded {len(self.pandas_df)} rows into pandas DataFrame")
            
            return self.pandas_df
        except Exception as e:
            print(f"❌ Error loading pandas DataFrame: {e}")
            return pd.DataFrame()
    
    def load_polars(self, nrows: Optional[int] = None) -> pl.DataFrame:
        """Load data as polars DataFrame with optional row limit."""
        try:
            if nrows:
                self.polars_df = pl.read_parquet(self.file_path).head(nrows)
                print(f"⚡ Loaded {len(self.polars_df)} rows (limited) into polars DataFrame")
            else:
                self.polars_df = pl.read_parquet(self.file_path)
                print(f"⚡ Loaded {len(self.polars_df)} rows into polars DataFrame")
            
            return self.polars_df
        except Exception as e:
            print(f"❌ Error loading polars DataFrame: {e}")
            return pl.DataFrame()
    
    def get_data_summary(self, library: str = "pandas", nrows: int = 1000) -> Dict[str, Any]:
        """Get statistical summary of the data."""
        if library == "pandas":
            if self.pandas_df is None:
                self.load_pandas(nrows)
            df = self.pandas_df
            
            summary = {
                "shape": df.shape,
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                "null_counts": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict(),
            }
            
            # Add describe for numeric columns
            numeric_cols = df.select_dtypes(include=[float, int]).columns
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
            
        elif library == "polars":
            if self.polars_df is None:
                self.load_polars(nrows)
            df = self.polars_df
            
            summary = {
                "shape": df.shape,
                "estimated_size_mb": round(df.estimated_size("mb"), 2),
                "null_counts": {col: df[col].null_count() for col in df.columns},
                "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            }
            
            # Add describe for numeric columns
            numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                          if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]]
            if len(numeric_cols) > 0:
                summary["numeric_summary"] = df.select(numeric_cols).describe().to_dict(as_series=False)
        
        return summary
    
    def preview_data(self, nrows: int = 10, library: str = "pandas"):
        """Preview the first few rows of data."""
        if library == "pandas":
            if self.pandas_df is None:
                self.load_pandas(nrows * 2)  # Load a bit more than preview
            print(f"\n📋 First {nrows} rows (pandas):")
            print(self.pandas_df.head(nrows))
            
        elif library == "polars":
            if self.polars_df is None:
                self.load_polars(nrows * 2)
            print(f"\n📋 First {nrows} rows (polars):")
            print(self.polars_df.head(nrows))
    
    def find_unique_values(self, column: str, limit: int = 20) -> List[Any]:
        """Find unique values in a specific column."""
        if self.pandas_df is not None:
            unique_vals = self.pandas_df[column].unique()
            return unique_vals[:limit].tolist() if len(unique_vals) > limit else unique_vals.tolist()
        elif self.polars_df is not None:
            unique_vals = self.polars_df[column].unique().to_list()
            return unique_vals[:limit] if len(unique_vals) > limit else unique_vals
        else:
            print("❌ No data loaded. Use load_pandas() or load_polars() first.")
            return []
    
    def search_columns(self, pattern: str) -> List[str]:
        """Search for columns matching a pattern."""
        if not self.file_info:
            return []
        
        schema = self.file_info.schema.to_arrow_schema()
        matching_cols = [field.name for field in schema if pattern.lower() in field.name.lower()]
        return matching_cols
    
    def export_summary_report(self, output_file: str = "data_summary_report.json"):
        """Export a comprehensive summary report to JSON."""
        report = {
            "file_info": self.get_basic_info(),
            "schema": self.get_schema_info(),
            "data_summary_pandas": self.get_data_summary("pandas", 1000),
            "data_summary_polars": self.get_data_summary("polars", 1000),
        }
        
        output_path = self.file_path.parent / output_file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Summary report exported to: {output_path}")
        return output_path


def explore_all_parquet_files(directory: str = ".") -> Dict[str, ParquetExplorer]:
    """Explore all parquet files in a directory."""
    directory_path = Path(directory)
    parquet_files = list(directory_path.glob("*.parquet"))
    
    if not parquet_files:
        print("❌ No parquet files found in the directory")
        return {}
    
    explorers = {}
    print(f"🔍 Found {len(parquet_files)} parquet file(s):")
    
    for file_path in parquet_files:
        print(f"\n📁 Processing: {file_path.name}")
        try:
            explorer = ParquetExplorer(str(file_path))
            explorers[file_path.name] = explorer
            
            # Print basic info
            info = explorer.get_basic_info()
            print(f"   📊 Rows: {info.get('num_rows', 'Unknown'):,}")
            print(f"   📋 Columns: {info.get('num_columns', 'Unknown')}")
            print(f"   💾 Size: {info.get('file_size_mb', 'Unknown')} MB")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path.name}: {e}")
    
    return explorers


if __name__ == "__main__":
    # Example usage
    print("🚀 IceCube Parquet Data Explorer")
    print("=" * 50)
    
    # Explore all parquet files in current directory
    explorers = explore_all_parquet_files()
    
    # If files were found, demonstrate detailed exploration
    if explorers:
        # Take the first file for detailed demonstration
        first_file = list(explorers.keys())[0]
        explorer = explorers[first_file]
        
        print(f"\n🔬 Detailed analysis of: {first_file}")
        print("-" * 40)
        
        # Show schema
        schema = explorer.get_schema_info()
        print(f"\n📋 Schema Information:")
        print(f"Total columns: {schema['total_columns']}")
        for col in schema['columns'][:10]:  # Show first 10 columns
            print(f"  - {col['name']}: {col['type']}")
        if len(schema['columns']) > 10:
            print(f"  ... and {len(schema['columns']) - 10} more columns")
        
        # Preview data
        explorer.preview_data(5, "pandas")
        
        # Export summary report
        explorer.export_summary_report()
