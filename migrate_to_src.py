#!/usr/bin/env python3
"""
XP3 PRO FOREX - Migration Script

This script automatically migrates legacy scripts to use the new src-layout architecture.
"""

import os
import re
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define migration mappings
IMPORT_MAPPINGS = {
    'import config_forex': 'from xp3_forex.core import config',
    'import utils_forex': 'from xp3_forex.utils import mt5_utils, indicators, calculations, data_utils',
    'from utils_forex': 'from xp3_forex.utils',
    'import news_filter': 'from xp3_forex.analysis import news_filter',
    'from news_filter': 'from xp3_forex.analysis.news_filter',
    'import validation_forex': 'from xp3_forex.risk import validation',
    'from validation_forex': 'from xp3_forex.risk.validation',
}

def migrate_file(file_path: Path) -> bool:
    """Migrate a single Python file to use new imports."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Add src path insertion if not present
        if 'sys.path.insert(0, str(Path(__file__).parent / "src"))' not in content:
            src_path_code = '''import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

'''
            # Find first import and insert before it
            first_import_match = re.search(r'^(import|from)\s+', content, re.MULTILINE)
            if first_import_match:
                insert_pos = first_import_match.start()
                content = content[:insert_pos] + src_path_code + content[insert_pos:]
        
        # Replace imports
        for old_import, new_import in IMPORT_MAPPINGS.items():
            content = content.replace(old_import, new_import)
        
        # Update specific function calls
        content = re.sub(r'utils_forex\.', 'mt5_utils.', content)
        content = re.sub(r'utils_forex\.', 'indicators.', content)
        content = re.sub(r'utils_forex\.', 'calculations.', content)
        content = re.sub(r'utils_forex\.', 'data_utils.', content)
        
        # Only write if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"✅ Migrated: {file_path}")
            return True
        else:
            logger.info(f"ℹ️ No changes needed: {file_path}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to migrate {file_path}: {e}")
        return False

def migrate_root_scripts():
    """Migrate all Python scripts in the root directory."""
    root_dir = Path(__file__).parent
    src_dir = root_dir / "src"
    
    # Scripts to migrate (excluding wrapper files)
    scripts_to_migrate = [
        'bot_forex.py',
        'utils_forex.py', 
        'config_forex.py',
        'validation_forex.py',
        'news_filter.py',
        'daily_analysis_logger.py',
        'fast_loop.py',
        'get_rates.py'
    ]
    
    migrated_count = 0
    
    for script_name in scripts_to_migrate:
        script_path = root_dir / script_name
        if script_path.exists():
            # Create backup
            backup_path = script_path.with_suffix('.py.backup')
            if not backup_path.exists():
                shutil.copy2(script_path, backup_path)
                logger.info(f"📋 Backup created: {backup_path}")
            
            if migrate_file(script_path):
                migrated_count += 1
        else:
            logger.warning(f"⚠️ Script not found: {script_name}")
    
    logger.info(f"🎯 Migration completed: {migrated_count} files migrated")
    return migrated_count

def create_migration_report():
    """Create a report of migration status."""
    root_dir = Path(__file__).parent
    report = []
    
    report.append("XP3 PRO FOREX - Migration Report")
    report.append("=" * 40)
    
    # Check src structure
    src_dir = root_dir / "src"
    if src_dir.exists():
        report.append(f"✅ src directory exists: {src_dir}")
        
        # Check subdirectories
        subdirs = ['xp3_forex', 'xp3_forex/core', 'xp3_forex/utils', 
                  'xp3_forex/strategies', 'xp3_forex/ml', 'xp3_forex/risk',
                  'xp3_forex/analysis', 'xp3_forex/optimization']
        
        for subdir in subdirs:
            subdir_path = src_dir / subdir
            if subdir_path.exists():
                report.append(f"✅ {subdir} exists")
            else:
                report.append(f"❌ {subdir} missing")
    else:
        report.append("❌ src directory missing")
    
    # Check wrapper files
    wrapper_files = [
        'bot_forex_wrapper.py',
        'utils_forex_wrapper.py', 
        'config_forex_wrapper.py'
    ]
    
    for wrapper in wrapper_files:
        wrapper_path = root_dir / wrapper
        if wrapper_path.exists():
            report.append(f"✅ {wrapper} exists")
        else:
            report.append(f"❌ {wrapper} missing")
    
    report.append("\nNext Steps:")
    report.append("1. Test the migrated scripts")
    report.append("2. Update any remaining hardcoded paths")
    report.append("3. Verify all imports work correctly")
    report.append("4. Run the bot with: python src/run_bot.py")
    
    report_content = "\n".join(report)
    
    with open(root_dir / "migration_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(report_content)
    return report_content

if __name__ == "__main__":
    print("🚀 XP3 PRO FOREX Migration Tool")
    print("=" * 40)
    
    # Migrate scripts
    migrated = migrate_root_scripts()
    
    # Create report
    print("\n" + "=" * 40)
    create_migration_report()
    
    print(f"\n✅ Migration tool completed! {migrated} files processed.")