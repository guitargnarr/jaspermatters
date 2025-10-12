#!/usr/bin/env python3
"""
GoDaddy DNS Updater for jaspermatters.com
Automatically configures DNS to point to GitHub Pages
"""

import requests
import json
import time
import os
from typing import List, Dict

# GitHub Pages IP addresses
GITHUB_IPS = [
    "185.199.108.153",
    "185.199.109.153", 
    "185.199.110.153",
    "185.199.111.153"
]

class GoDaddyDNSUpdater:
    def __init__(self, api_key: str, api_secret: str, domain: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.domain = domain
        self.base_url = "https://api.godaddy.com/v1"
        self.headers = {
            "Authorization": f"sso-key {api_key}:{api_secret}",
            "Content-Type": "application/json"
        }
    
    def get_current_records(self) -> List[Dict]:
        """Fetch current DNS records"""
        url = f"{self.base_url}/domains/{self.domain}/records"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching records: {e}")
            return []
    
    def update_dns_records(self) -> bool:
        """Update DNS records to point to GitHub Pages"""
        
        # Get current records
        print(f"🔍 Fetching current DNS records for {self.domain}...")
        current_records = self.get_current_records()
        
        if not current_records:
            print("❌ Could not fetch current records. Check API credentials.")
            return False
        
        # Filter out old A records for @ and www CNAME
        new_records = []
        for record in current_records:
            # Keep all records except A records for @ and CNAME for www
            if not ((record['type'] == 'A' and record['name'] == '@') or 
                   (record['type'] == 'CNAME' and record['name'] == 'www')):
                new_records.append(record)
        
        # Add GitHub Pages A records
        for ip in GITHUB_IPS:
            new_records.append({
                "type": "A",
                "name": "@",
                "data": ip,
                "ttl": 600
            })
        
        # Add www CNAME
        new_records.append({
            "type": "CNAME",
            "name": "www",
            "data": "guitargnar.github.io",
            "ttl": 600
        })
        
        # Update DNS records
        print(f"📝 Updating DNS records...")
        url = f"{self.base_url}/domains/{self.domain}/records"
        
        try:
            response = requests.put(url, headers=self.headers, json=new_records)
            response.raise_for_status()
            print(f"✅ DNS records updated successfully!")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Error updating records: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Response: {e.response.text}")
            return False
    
    def verify_dns(self) -> bool:
        """Verify DNS points to GitHub Pages"""
        print(f"\n🔍 Verifying DNS configuration...")
        
        import socket
        try:
            # Check A records
            ips = socket.gethostbyname_ex(self.domain)[2]
            github_ips_found = [ip for ip in ips if ip in GITHUB_IPS]
            
            if github_ips_found:
                print(f"✅ Domain points to GitHub Pages IPs: {github_ips_found}")
                return True
            else:
                print(f"⚠️  Domain currently points to: {ips}")
                print(f"    Expected GitHub IPs: {GITHUB_IPS}")
                return False
        except socket.gaierror as e:
            print(f"❌ Could not resolve domain: {e}")
            return False

def main():
    print("🚀 GoDaddy DNS Updater for GitHub Pages")
    print("=" * 50)
    
    # Check for API credentials
    api_key = os.getenv('GODADDY_API_KEY')
    api_secret = os.getenv('GODADDY_API_SECRET')
    
    if not api_key or not api_secret:
        print("\n⚠️  GoDaddy API credentials not found!")
        print("\nTo get your API credentials:")
        print("1. Go to: https://developer.godaddy.com/keys")
        print("2. Create a production API key")
        print("3. Set environment variables:")
        print("   export GODADDY_API_KEY='your_key_here'")
        print("   export GODADDY_API_SECRET='your_secret_here'")
        print("\nThen run this script again.")
        
        print("\n" + "=" * 50)
        print("📋 MANUAL STEPS (if you prefer):")
        print("\n1. Go to GoDaddy DNS management")
        print("2. DELETE these A records:")
        print("   - @ → 18.208.88.157")
        print("   - @ → 98.84.224.111")
        print("\n3. ADD these A records:")
        for ip in GITHUB_IPS:
            print(f"   - @ → {ip}")
        print("\n4. ADD this CNAME:")
        print("   - www → guitargnar.github.io")
        return
    
    # Initialize updater
    updater = GoDaddyDNSUpdater(api_key, api_secret, "jaspermatters.com")
    
    # Show current configuration
    current = updater.get_current_records()
    if current:
        print("\n📋 Current A records:")
        for record in current:
            if record['type'] == 'A' and record['name'] == '@':
                print(f"   - {record['data']}")
    
    # Update DNS
    print("\n🔄 Updating to GitHub Pages configuration...")
    if updater.update_dns_records():
        print("\n⏳ Waiting for DNS propagation (this may take 5-30 minutes)...")
        
        # Check DNS after a short wait
        time.sleep(10)
        updater.verify_dns()
        
        print("\n✅ Next Steps:")
        print("1. Go to GitHub Pages settings")
        print("2. Click 'Check again' next to DNS error")
        print("3. Once verified, enable 'Enforce HTTPS'")
        print("4. Your ML platform will be live at https://jaspermatters.com!")
    else:
        print("\n❌ DNS update failed. Please update manually in GoDaddy.")

if __name__ == "__main__":
    main()