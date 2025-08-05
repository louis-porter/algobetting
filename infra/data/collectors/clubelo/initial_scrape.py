import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClubEloScraper:
    def __init__(self, db_path=r'C:\Users\Owner\dev\algobetting\infra\data\db\algobetting.db'):
        self.db_path = db_path
        self.base_url = 'http://api.clubelo.com/'
        self.session = requests.Session()
        # Set session headers and timeout defaults
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def get_all_tuesdays(self, start_date, end_date):
        """Generate all Tuesday dates between start_date and end_date"""
        tuesdays = []
        current = start_date
        
        # Find the first Tuesday
        while current.weekday() != 1:  # Tuesday is 1 (Monday=0)
            current += timedelta(days=1)
        
        # Collect all Tuesdays
        while current <= end_date:
            tuesdays.append(current)
            current += timedelta(days=7)
            
        return tuesdays
    
    def fetch_csv_data(self, date_str):
        """Fetch CSV data from the API for a given date"""
        url = f"{self.base_url}{date_str}"
        
        try:
            # Try with a shorter timeout and specific headers
            response = self.session.get(
                url, 
                timeout=(10, 30),  # (connect timeout, read timeout)
                headers={'Accept': 'text/csv, */*'}
            )
            response.raise_for_status()
            
            # Check if we got data
            if response.text and len(response.text) > 0:
                logger.info(f"Successfully fetched data for {date_str} ({len(response.text)} bytes)")
                return response.text
            else:
                logger.warning(f"Empty response for {date_str}")
                return None
                
        except requests.exceptions.ConnectTimeout as e:
            logger.error(f"Connection timeout for {date_str}: {e}")
            return None
        except requests.exceptions.ReadTimeout as e:
            logger.error(f"Read timeout for {date_str}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data for {date_str}: {e}")
            return None
    
    def csv_to_dataframe(self, csv_data, date_str):
        """Convert CSV string to pandas DataFrame"""
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(csv_data))
            
            # Add the date column
            df['fetch_date'] = date_str
            
            return df
        except Exception as e:
            logger.error(f"Error parsing CSV for {date_str}: {e}")
            return None
    
    def setup_database(self):
        """Create the database table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # We'll create the table dynamically based on the first successful CSV
        # For now, just ensure the database file exists
        conn.close()
    
    def create_table_from_dataframe(self, df, table_name='clubelo_ratings_raw'):
        """Create table based on DataFrame structure"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Drop table if exists and create new one
            # You might want to change this to IF NOT EXISTS for production
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Created table '{table_name}' with {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error creating table: {e}")
        finally:
            conn.close()
    
    def insert_dataframe_to_db(self, df, table_name='clubelo_ratings_raw'):
        """Insert DataFrame data into existing table"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df.to_sql(table_name, conn, if_exists='append', index=False)
            logger.info(f"Inserted {len(df)} rows for date {df['fetch_date'].iloc[0]}")
            
        except Exception as e:
            logger.error(f"Error inserting data: {e}")
        finally:
            conn.close()
    
    def date_already_exists(self, date_str, table_name='clubelo_ratings_raw'):
        """Check if data for a specific date already exists"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE fetch_date = ?", (date_str,))
            count = cursor.fetchone()[0]
            return count > 0
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return False
        finally:
            conn.close()
    
    def run_scraper(self, start_date_str='2017-01-01', delay_seconds=2, max_retries=3):
        """Main method to run the scraper"""
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.now()
        
        logger.info(f"Starting scraper from {start_date_str} to {end_date.strftime('%Y-%m-%d')}")
        
        # Get all Tuesday dates
        tuesdays = self.get_all_tuesdays(start_date, end_date)
        logger.info(f"Found {len(tuesdays)} Tuesdays to process")
        
        # Setup database
        self.setup_database()
        
        table_created = False
        processed_count = 0
        skipped_count = 0
        failed_count = 0
        
        for i, tuesday in enumerate(tuesdays, 1):
            date_str = tuesday.strftime('%Y-%m-%d')
            
            # Skip if already exists
            if table_created and self.date_already_exists(date_str):
                logger.info(f"Data for {date_str} already exists, skipping")
                skipped_count += 1
                continue
            
            logger.info(f"Processing {date_str} ({i}/{len(tuesdays)})")
            
            # Try fetching with retries
            csv_data = None
            for attempt in range(max_retries):
                csv_data = self.fetch_csv_data(date_str)
                if csv_data:
                    break
                elif attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {date_str}")
                    time.sleep(delay_seconds * 2)  # Longer delay on retry
            
            if csv_data:
                # Convert to DataFrame
                df = self.csv_to_dataframe(csv_data, date_str)
                
                if df is not None and not df.empty:
                    if not table_created:
                        # Create table with first successful data
                        self.create_table_from_dataframe(df)
                        table_created = True
                    else:
                        # Insert into existing table
                        self.insert_dataframe_to_db(df)
                    
                    processed_count += 1
                else:
                    logger.warning(f"Empty or invalid data for {date_str}")
                    failed_count += 1
            else:
                logger.error(f"Failed to fetch data for {date_str} after {max_retries} attempts")
                failed_count += 1
            
            # Add delay to be respectful to the API
            time.sleep(delay_seconds)
        
        logger.info(f"Scraping completed. Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}")
        
        # Show summary
        if table_created:
            self.show_summary()
    
    def show_summary(self, table_name='clubelo_ratings_raw'):
        """Show summary of data in the database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            df = pd.read_sql_query(f"SELECT fetch_date, COUNT(*) as record_count FROM {table_name} GROUP BY fetch_date ORDER BY fetch_date", conn)
            logger.info(f"Database summary:\n{df.to_string(index=False)}")
            
            total_records = pd.read_sql_query(f"SELECT COUNT(*) as total FROM {table_name}", conn)
            logger.info(f"Total records in database: {total_records['total'].iloc[0]}")
            
        except Exception as e:
            logger.error(f"Error showing summary: {e}")
        finally:
            conn.close()

def main():
    """Main function to run the scraper"""
    scraper = ClubEloScraper()
    
    # Run the scraper
    # You can customize the start date, delay between requests, and retry attempts
    scraper.run_scraper(start_date_str='2017-01-01', delay_seconds=2, max_retries=3)

if __name__ == "__main__":
    main()