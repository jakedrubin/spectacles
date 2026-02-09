"""scraper.py
Full-featured web scraper for extracting eyeglass product information from americasbest.com.

This module provides the AmericasBestScraper class which:
- Discovers product pages by crawling through paginated listing pages
- Extracts detailed product specifications (brand, gender, frame material, shape, type, SKU, price)
- Uses multiple extraction strategies (HTML tables, structured lists, data attributes, JSON-LD)
- Handles errors with retries and exponential backoff
- Exports scraped data to JSON and CSV formats

Usage:
    As a script:
        python scraper.py
    
    As a module:
        from scraper import AmericasBestScraper
        scraper = AmericasBestScraper()
        data = scraper.scrape_all_glasses()
        scraper.save_to_json()
        scraper.save_to_csv()

Requirements:
    pip install requests beautifulsoup4 lxml
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import re
import os
from urllib.parse import urljoin, urlparse
import logging

class AmericasBestScraper:
    def __init__(self):
        self.base_url = "https://www.americasbest.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.glasses_data = []
        
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_page(self, url, retries=3):
        """Fetch a page with error handling and retries"""
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None

    def find_glasses_pages(self):
        """Find all individual glasses product pages from listing pages"""
        glasses_urls = set()  # Use set to avoid duplicates
        
        # Start with the main listing page
        base_listing_url = f"{self.base_url}/all-glasses/c/100"
        page_num = 0
        
        while True:
            # Construct URL for current page
            if page_num == 0:
                current_url = f"{base_listing_url}?sort=relevance"
            else:
                current_url = f"{base_listing_url}?q=%3Arelevance&page={page_num}"
            
            self.logger.info(f"Scraping listing page: {current_url}")
            response = self.get_page(current_url)
            if not response:
                break
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find product links - they follow pattern /product-name/p/product-id
            product_links = soup.find_all('a', href=True)
            products_found = 0
            
            for link in product_links:
                href = link.get('href')
                if href and '/p/' in href and not href.startswith('http'):
                    # Convert relative URL to absolute
                    full_url = urljoin(self.base_url, href)
                    # Filter out non-product URLs
                    if '/p/' in full_url and len(href.split('/p/')) == 2:
                        glasses_urls.add(full_url)
                        products_found += 1
            
            self.logger.info(f"Found {products_found} products on page {page_num + 1}")
            
            # Check if there are more pages by looking for pagination or "Load More"
            # Look for next page indicators
            has_next_page = False
            
            # Check for pagination links
            pagination_links = soup.find_all('a', class_=['next', 'pagination-next'])
            if pagination_links:
                has_next_page = True
            
            # Check for "Load More" or similar buttons (common in modern e-commerce)
            load_more = soup.find_all(['button', 'a'], text=lambda x: x and ('load more' in x.lower() or 'show more' in x.lower()))
            if load_more:
                has_next_page = True
            
            # If no products found on this page, likely reached the end
            if products_found == 0:
                self.logger.info("No products found on this page, stopping pagination")
                break
            
            # If no next page indicators and we found products, try next page anyway (up to reasonable limit)
            if not has_next_page and page_num < 50:  # Safety limit
                page_num += 1
                continue
            elif has_next_page:
                page_num += 1
            else:
                break

            time.sleep(2)  # Be respectful with requests
        
        glasses_urls_list = list(glasses_urls)
        self.logger.info(f"Total unique product URLs found: {len(glasses_urls_list)}")
        return glasses_urls_list

    def scrape_glasses_details(self, url):
        """Scrape detailed information from a glasses product page"""
        response = self.get_page(url)
        if not response:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        glasses_info = {
            'url': url,
            'name': '',
            'brand': '',
            'gender': '',
            'frame_material': '',
            'frame_shape': '',
            'frame_type': '',
            'sku': '',
            'price': ''
        }
        
        # Extract specs information - try multiple approaches
        self.extract_specs_from_table(soup, glasses_info)
        self.extract_specs_from_list(soup, glasses_info)
        self.extract_specs_from_data_attributes(soup, glasses_info)
        self.extract_specs_from_json_ld(soup, glasses_info)
        
        # Extract basic product info if not found in specs
        if not glasses_info['name']:
            name_selectors = ['h1.product-name', 'h1', '.product-title', '[data-testid="product-name"]']
            for selector in name_selectors:
                name_elem = soup.select_one(selector)
                if name_elem:
                    glasses_info['name'] = name_elem.get_text(strip=True)
                    break
        
        # Extract price with better logic
        self.extract_price_info(soup, glasses_info)
        
        # Extract description
        desc_selectors = ['.description', '.product-description', '[data-testid="description"]']
        for selector in desc_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                glasses_info['description'] = desc_elem.get_text(strip=True)
                break
        
        return glasses_info

    def extract_specs_from_json_ld(self, soup, glasses_info):
        """Extract specs from JSON-LD structured data"""
        json_scripts = soup.find_all('script', type='application/ld+json')
        for script in json_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    # Check for product information
                    if data.get('@type') == 'Product':
                        if not glasses_info['name'] and data.get('name'):
                            glasses_info['name'] = data['name']
                        if not glasses_info['brand'] and data.get('brand'):
                            brand = data['brand']
                            if isinstance(brand, dict):
                                glasses_info['brand'] = brand.get('name', '')
                            else:
                                glasses_info['brand'] = str(brand)
                        if not glasses_info['sku'] and data.get('sku'):
                            glasses_info['sku'] = data['sku']
            except json.JSONDecodeError:
                continue

    def extract_specs_from_data_attributes(self, soup, glasses_info):
        """Extract specs from data attributes"""
        # Look for elements with data attributes
        for elem in soup.find_all(attrs={'data-brand': True}):
            glasses_info['brand'] = elem.get('data-brand')
        for elem in soup.find_all(attrs={'data-gender': True}):
            glasses_info['gender'] = elem.get('data-gender')
        for elem in soup.find_all(attrs={'data-material': True}):
            glasses_info['frame_material'] = elem.get('data-material')
        for elem in soup.find_all(attrs={'data-shape': True}):
            glasses_info['frame_shape'] = elem.get('data-shape')
        for elem in soup.find_all(attrs={'data-type': True}):
            glasses_info['frame_type'] = elem.get('data-type')
        for elem in soup.find_all(attrs={'data-sku': True}):
            glasses_info['sku'] = elem.get('data-sku')
        
        # Also check for product ID in URL as backup SKU
        if not glasses_info['sku']:
            url_parts = glasses_info['url'].split('/p/')
            if len(url_parts) > 1:
                glasses_info['sku'] = url_parts[1].split('?')[0].split('#')[0]

    def extract_specs_from_table(self, soup, glasses_info):
        """Extract specs from a table format"""
        # Look for specs table with more specific selectors
        specs_tables = soup.find_all('table', class_=['specs', 'specifications', 'product-specs', 'product-details'])
        if not specs_tables:
            # Look for any table that might contain specs
            all_tables = soup.find_all('table')
            specs_tables = [table for table in all_tables if 
                          any(keyword in table.get_text().lower() for keyword in ['brand', 'material', 'frame', 'gender'])]
        
        for table in specs_tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    key = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    self.map_spec_value(key, value, glasses_info)

    def extract_specs_from_list(self, soup, glasses_info):
        """Extract specs from structured list format on product pages"""
        spec_items = soup.select('.m-product-specs__item')

        for item in spec_items:
            label_elem = item.select_one('.m-product-specs__item-label strong')
            value_elem = item.select_one('.m-product-specs__item-value')

            if not label_elem or not value_elem:
                # Check alternative SKU structure
                label_elem = item.select_one('strong')
                value_elem = item.select_one('.m-product-specs__item-value--product-code')

            if label_elem and value_elem:
                key = label_elem.get_text(strip=True).lower()
                value = value_elem.get_text(strip=True)
                self.map_spec_value(key, value, glasses_info)


        def parse_americas_best_specs(self, container, glasses_info):
            """Parse Americas Best specific specs format"""
            # Get all text and look for the pattern
            text = container.get_text()
            self.extract_specs_from_text_pattern(text, glasses_info)
            
            # Also look for links that might contain spec values
            links = container.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                link_text = link.get_text(strip=True)
                
                # Brand links typically go to brand pages
                if '/archer' in href.lower() or 'archer' in link_text.lower():
                    if not glasses_info['brand']:
                        glasses_info['brand'] = link_text
                elif 'women' in href.lower() or 'men' in href.lower():
                    if not glasses_info['gender']:
                        glasses_info['gender'] = link_text
                elif 'plastic' in href.lower() or 'metal' in href.lower():
                    if not glasses_info['frame_material']:
                        glasses_info['frame_material'] = link_text
                elif 'full-rim' in href.lower() or 'rimless' in href.lower():
                    if not glasses_info['frame_type']:
                        glasses_info['frame_type'] = link_text

    def parse_specs_from_page_text(self, soup, glasses_info):
        """Parse specs from the entire page text using Americas Best pattern"""
        page_text = soup.get_text()
        self.extract_specs_from_text_pattern(page_text, glasses_info)

    def extract_specs_from_text_pattern(self, text, glasses_info):
        """Extract specs from Americas Best text pattern"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for "Brand" followed by actual brand name
            if line == "Brand" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line != "-" and not glasses_info['brand']:
                    glasses_info['brand'] = next_line
            
            # Look for "Gender" 
            elif line == "Gender" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line != "-" and not glasses_info['gender']:
                    glasses_info['gender'] = next_line
            
            # Look for "Frame Material"
            elif line == "Frame Material" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line != "-" and not glasses_info['frame_material']:
                    glasses_info['frame_material'] = next_line
            
            # Look for "Frame Shape"
            elif line == "Frame Shape" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line != "-" and not glasses_info['frame_shape']:
                    glasses_info['frame_shape'] = next_line
            
            # Look for "Frame Type"
            elif line == "Frame Type" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and next_line != "-" and not glasses_info['frame_type']:
                    glasses_info['frame_type'] = next_line
            
            # Look for SKU pattern
            elif "SKU" in line:
                # Extract number after SKU
                sku_match = line.split("SKU")[-1].strip()
                if sku_match and not glasses_info['sku']:
                    glasses_info['sku'] = sku_match

    def extract_price_info(self, soup, glasses_info):
        """Extract clean price information"""
        # Look for price patterns
        price_selectors = [
            '.price-current',
            '.current-price', 
            '.product-price',
            '[data-price]',
            '.price'
        ]
        
        for selector in price_selectors:
            price_elem = soup.select_one(selector)
            if price_elem:
                price_text = price_elem.get_text(strip=True)
                # Clean up price text
                if '$' in price_text:
                    # Extract just the price, not ranges or other text
                    price_parts = price_text.split('$')
                    for part in price_parts[1:]:  # Skip first empty part
                        clean_price = part.split()[0]  # Get first part before space
                        try:
                            # Validate it's a proper price
                            float(clean_price.replace(',', ''))
                            glasses_info['price'] = f"${clean_price}"
                            return
                        except ValueError:
                            continue
        
        # Fallback: look in page text for price patterns
        page_text = soup.get_text()
        price_matches = re.findall(r'\$(\d+\.?\d*)', page_text)
        if price_matches:
            # Take the first reasonable price (between $20-$500)
            for price in price_matches:
                try:
                    price_val = float(price)
                    if 20 <= price_val <= 500:
                        glasses_info['price'] = f"${price}"
                        break
                except ValueError:
                    continue

    def map_spec_value(self, key, value, glasses_info):
        """Map extracted key-value pairs to the appropriate glasses_info field"""
        key = key.lower().strip().rstrip(':')
        value = value.strip()
        
        if not value:  # Skip empty values
            return
        
        # Brand mapping
        if any(brand_key in key for brand_key in ['brand', 'manufacturer', 'designer', 'make']):
            if not glasses_info['brand']:
                glasses_info['brand'] = value
        
        # Gender mapping
        elif any(gender_key in key for gender_key in ['gender', 'sex', 'for', 'mens', 'womens', 'unisex']):
            if not glasses_info['gender']:
                glasses_info['gender'] = value
        
        # Frame material mapping
        elif any(material_key in key for material_key in ['material', 'frame material', 'construction', 'made of']):
            if not glasses_info['frame_material']:
                glasses_info['frame_material'] = value
        
        # Frame shape mapping
        elif any(shape_key in key for shape_key in ['shape', 'frame shape', 'style', 'silhouette']):
            if not glasses_info['frame_shape']:
                glasses_info['frame_shape'] = value
        
        # Frame type mapping
        elif any(type_key in key for type_key in ['type', 'frame type', 'rim', 'rim type', 'construction type']):
            if not glasses_info['frame_type']:
                glasses_info['frame_type'] = value
        
        # SKU mapping
        elif any(sku_key in key for sku_key in ['sku', 'model', 'product id', 'item number', 'model number', 'style number']):
            if not glasses_info['sku']:
                glasses_info['sku'] = value

    def scrape_all_glasses(self):
        """Main method to scrape all glasses information"""
        self.logger.info("Starting to find all glasses pages...")
        glasses_urls = self.find_glasses_pages()
        self.logger.info(f"Found {len(glasses_urls)} glasses URLs")
        
        for i, url in enumerate(glasses_urls, 1):
            self.logger.info(f"Scraping {i}/{len(glasses_urls)}: {url}")
            
            glasses_info = self.scrape_glasses_details(url)
            if glasses_info:
                self.glasses_data.append(glasses_info)
            
            # Be respectful with request timing
            time.sleep(2)
        
        self.logger.info(f"Scraped {len(self.glasses_data)} glasses successfully")
        return self.glasses_data

    def save_to_json(self, filename='americas_best_glasses.json'):
        """Save scraped data to JSON file"""
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.glasses_data, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Data saved to {filepath}")

    def save_to_csv(self, filename='americas_best_glasses.csv'):
        """Save scraped data to CSV file"""
        if not self.glasses_data:
            self.logger.warning("No data to save")
            return
        
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        filepath = os.path.join(data_dir, filename)
        fieldnames = self.glasses_data[0].keys()
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for glasses in self.glasses_data:
                # Convert lists to strings for CSV
                row = glasses.copy()
                for key, value in row.items():
                    if isinstance(value, list):
                        row[key] = '; '.join(map(str, value))
                writer.writerow(row)
        self.logger.info(f"Data saved to {filepath}")

def main():
    """Main execution function"""
    scraper = AmericasBestScraper()
    
    # Check robots.txt first
    print("Please check https://www.americasbest.com/robots.txt before running this scraper")
    print("Make sure you comply with their terms of service")
    
    # Allow bypassing the interactive prompt by setting environment variable SCRAPE_CONFIRM=1
    import os
    if os.environ.get('SCRAPE_CONFIRM') == '1':
        proceed = True
    else:
        try:
            confirm = input("Do you want to proceed? (y/n): ")
            proceed = confirm.lower() == 'y'
        except Exception:
            # In non-interactive environments default to not proceeding
            proceed = False
    if not proceed:
        print("Scraping cancelled")
        return
    
    try:
        # Scrape all glasses
        glasses_data = scraper.scrape_all_glasses()
        
        # Save data
        scraper.save_to_json()
        scraper.save_to_csv()
        
        print(f"Successfully scraped {len(glasses_data)} glasses")
        
    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Required packages: requests, beautifulsoup4, lxml
    # Install with: pip install requests beautifulsoup4 lxml
    main()