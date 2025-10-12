"""
Job Scraping Pipeline
Collects AI/ML job postings from multiple sources
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Optional
import time
import json
from datetime import datetime
import re
import asyncio
import aiohttp
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobPosting:
    """Structured job posting data"""
    job_id: str
    title: str
    company: str
    location: str
    description: str
    requirements: List[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    job_type: str  # full-time, contract, etc
    remote: bool
    posted_date: datetime
    source: str
    url: str
    seniority_level: Optional[str]
    
    def to_dict(self):
        data = asdict(self)
        data['posted_date'] = self.posted_date.isoformat()
        return data


class JobScraper:
    """Multi-source job scraping system"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.jobs = []
        
    def extract_salary(self, text: str) -> tuple[Optional[float], Optional[float]]:
        """Extract salary range from text"""
        if not text:
            return None, None
            
        # Common salary patterns
        patterns = [
            r'\$(\d+)[kK]\s*-\s*\$(\d+)[kK]',  # $100k - $150k
            r'\$(\d+),?(\d+)\s*-\s*\$(\d+),?(\d+)',  # $100,000 - $150,000
            r'(\d+)[kK]\s*-\s*(\d+)[kK]',  # 100k - 150k
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return float(groups[0]) * 1000, float(groups[1]) * 1000
                elif len(groups) == 4:
                    min_sal = float(groups[0] + groups[1] if groups[1] else groups[0])
                    max_sal = float(groups[2] + groups[3] if groups[3] else groups[2])
                    return min_sal, max_sal
                    
        return None, None
    
    def extract_requirements(self, description: str) -> List[str]:
        """Extract skill requirements from job description"""
        requirements = []
        
        # Common requirement patterns
        skill_keywords = [
            'Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras',
            'SQL', 'NoSQL', 'MongoDB', 'PostgreSQL', 'Redis',
            'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'Statistics', 'Mathematics', 'Linear Algebra',
            'Git', 'CI/CD', 'Agile', 'Scrum',
            'Spark', 'Hadoop', 'Kafka', 'Airflow',
            'REST API', 'GraphQL', 'Microservices',
            'JavaScript', 'React', 'Node.js', 'FastAPI', 'Flask'
        ]
        
        description_lower = description.lower()
        for skill in skill_keywords:
            if skill.lower() in description_lower:
                requirements.append(skill)
                
        return list(set(requirements))
    
    def determine_seniority(self, title: str, description: str) -> str:
        """Determine seniority level from title and description"""
        title_lower = title.lower()
        desc_lower = description.lower()
        
        if any(word in title_lower for word in ['junior', 'jr', 'entry', 'associate']):
            return 'Junior'
        elif any(word in title_lower for word in ['senior', 'sr', 'lead', 'principal', 'staff']):
            return 'Senior'
        elif any(word in title_lower for word in ['manager', 'director', 'vp', 'head']):
            return 'Management'
        elif '0-2 years' in desc_lower or 'entry level' in desc_lower:
            return 'Junior'
        elif '5+ years' in desc_lower or '7+ years' in desc_lower or '10+ years' in desc_lower:
            return 'Senior'
        else:
            return 'Mid-level'
    
    async def scrape_indeed(self, query: str = "machine learning engineer", location: str = "Remote", pages: int = 5):
        """Scrape jobs from Indeed"""
        logger.info(f"Scraping Indeed for '{query}' in '{location}'")
        
        base_url = "https://www.indeed.com/jobs"
        jobs_found = []
        
        for page in range(pages):
            params = {
                'q': query,
                'l': location,
                'start': page * 10,
                'sort': 'date'
            }
            
            try:
                response = self.session.get(base_url, params=params)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find job cards
                job_cards = soup.find_all('div', class_='job_seen_beacon') or \
                           soup.find_all('div', class_='jobsearch-SerpJobCard')
                
                for card in job_cards:
                    try:
                        # Extract job details
                        title_elem = card.find('h2', class_='jobTitle') or card.find('a', class_='jobtitle')
                        company_elem = card.find('span', class_='companyName') or card.find('span', class_='company')
                        location_elem = card.find('div', class_='companyLocation') or card.find('span', class_='location')
                        description_elem = card.find('div', class_='job-snippet')
                        
                        if not all([title_elem, company_elem]):
                            continue
                            
                        title = title_elem.text.strip()
                        company = company_elem.text.strip()
                        location = location_elem.text.strip() if location_elem else "Remote"
                        description = description_elem.text.strip() if description_elem else ""
                        
                        # Check for salary
                        salary_elem = card.find('div', class_='salary-snippet')
                        salary_text = salary_elem.text.strip() if salary_elem else ""
                        salary_min, salary_max = self.extract_salary(salary_text)
                        
                        # Create job posting
                        job = JobPosting(
                            job_id=f"indeed_{hash(title + company)}",
                            title=title,
                            company=company,
                            location=location,
                            description=description,
                            requirements=self.extract_requirements(description),
                            salary_min=salary_min,
                            salary_max=salary_max,
                            job_type="Full-time",
                            remote="remote" in location.lower(),
                            posted_date=datetime.now(),
                            source="Indeed",
                            url=f"https://www.indeed.com/viewjob?jk={hash(title + company)}",
                            seniority_level=self.determine_seniority(title, description)
                        )
                        
                        jobs_found.append(job)
                        
                    except Exception as e:
                        logger.error(f"Error parsing job card: {e}")
                        continue
                
                # Rate limiting
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error scraping Indeed page {page}: {e}")
                continue
                
        logger.info(f"Found {len(jobs_found)} jobs from Indeed")
        self.jobs.extend(jobs_found)
        return jobs_found
    
    def scrape_linkedin_rss(self, query: str = "machine learning", location: str = "United States"):
        """Scrape LinkedIn jobs via RSS (no auth required)"""
        logger.info(f"Scraping LinkedIn RSS for '{query}'")
        
        # LinkedIn RSS feed for job searches
        rss_url = f"https://www.linkedin.com/jobs/rss/{query}-jobs-{location}"
        
        try:
            response = self.session.get(rss_url)
            soup = BeautifulSoup(response.content, 'xml')
            
            items = soup.find_all('item')
            jobs_found = []
            
            for item in items[:50]:  # Limit to 50 jobs
                try:
                    title = item.find('title').text
                    company = item.find('company').text if item.find('company') else "Unknown"
                    description = item.find('description').text
                    link = item.find('link').text
                    pub_date = item.find('pubDate').text
                    
                    # Clean description
                    desc_soup = BeautifulSoup(description, 'html.parser')
                    clean_desc = desc_soup.get_text()
                    
                    job = JobPosting(
                        job_id=f"linkedin_{hash(link)}",
                        title=title,
                        company=company,
                        location=location,
                        description=clean_desc[:1000],  # Limit description length
                        requirements=self.extract_requirements(clean_desc),
                        salary_min=None,
                        salary_max=None,
                        job_type="Full-time",
                        remote="remote" in title.lower() or "remote" in clean_desc.lower(),
                        posted_date=datetime.now(),
                        source="LinkedIn",
                        url=link,
                        seniority_level=self.determine_seniority(title, clean_desc)
                    )
                    
                    jobs_found.append(job)
                    
                except Exception as e:
                    logger.error(f"Error parsing LinkedIn job: {e}")
                    continue
                    
            logger.info(f"Found {len(jobs_found)} jobs from LinkedIn")
            self.jobs.extend(jobs_found)
            return jobs_found
            
        except Exception as e:
            logger.error(f"Error scraping LinkedIn RSS: {e}")
            return []
    
    def scrape_remoteok(self):
        """Scrape RemoteOK for remote ML/AI jobs"""
        logger.info("Scraping RemoteOK for ML/AI jobs")
        
        url = "https://remoteok.io/api"
        
        try:
            response = self.session.get(url)
            jobs_data = response.json()
            
            jobs_found = []
            ml_keywords = ['machine learning', 'ml', 'ai', 'data science', 'deep learning']
            
            for job_data in jobs_data[1:51]:  # Skip first item (metadata), limit to 50
                try:
                    # Filter for ML/AI jobs
                    position = job_data.get('position', '').lower()
                    tags = ' '.join(job_data.get('tags', [])).lower()
                    
                    if not any(keyword in position or keyword in tags for keyword in ml_keywords):
                        continue
                    
                    # Extract salary if available
                    salary_min = job_data.get('salary_min')
                    salary_max = job_data.get('salary_max')
                    
                    job = JobPosting(
                        job_id=f"remoteok_{job_data.get('id')}",
                        title=job_data.get('position'),
                        company=job_data.get('company'),
                        location="Remote",
                        description=job_data.get('description', ''),
                        requirements=job_data.get('tags', []),
                        salary_min=salary_min,
                        salary_max=salary_max,
                        job_type="Full-time",
                        remote=True,
                        posted_date=datetime.fromtimestamp(job_data.get('epoch', 0)),
                        source="RemoteOK",
                        url=job_data.get('url', ''),
                        seniority_level=self.determine_seniority(
                            job_data.get('position', ''), 
                            job_data.get('description', '')
                        )
                    )
                    
                    jobs_found.append(job)
                    
                except Exception as e:
                    logger.error(f"Error parsing RemoteOK job: {e}")
                    continue
                    
            logger.info(f"Found {len(jobs_found)} ML/AI jobs from RemoteOK")
            self.jobs.extend(jobs_found)
            return jobs_found
            
        except Exception as e:
            logger.error(f"Error scraping RemoteOK: {e}")
            return []
    
    def save_jobs(self, filename: str = "jobs_data.json"):
        """Save scraped jobs to JSON file"""
        logger.info(f"Saving {len(self.jobs)} jobs to {filename}")
        
        jobs_data = [job.to_dict() for job in self.jobs]
        
        with open(filename, 'w') as f:
            json.dump(jobs_data, f, indent=2)
            
        # Also save as DataFrame for ML processing
        df = pd.DataFrame(jobs_data)
        df.to_csv(filename.replace('.json', '.csv'), index=False)
        
        logger.info(f"Jobs saved to {filename} and {filename.replace('.json', '.csv')}")
        return df
    
    async def scrape_all(self):
        """Scrape all sources"""
        logger.info("Starting comprehensive job scraping...")
        
        # Scrape Indeed
        await self.scrape_indeed("machine learning engineer", "Remote", pages=3)
        await self.scrape_indeed("data scientist", "Remote", pages=2)
        await self.scrape_indeed("AI engineer", "Remote", pages=2)
        
        # Scrape LinkedIn
        self.scrape_linkedin_rss("machine-learning", "united-states")
        self.scrape_linkedin_rss("artificial-intelligence", "united-states")
        
        # Scrape RemoteOK
        self.scrape_remoteok()
        
        # Remove duplicates
        unique_jobs = {}
        for job in self.jobs:
            key = f"{job.title}_{job.company}"
            if key not in unique_jobs:
                unique_jobs[key] = job
                
        self.jobs = list(unique_jobs.values())
        
        logger.info(f"Total unique jobs collected: {len(self.jobs)}")
        
        # Save to files
        import os
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml", "data", "jobs_data.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df = self.save_jobs(save_path)
        
        return df


if __name__ == "__main__":
    scraper = JobScraper()
    
    # Run the scraper
    asyncio.run(scraper.scrape_all())
    
    print(f"\n✅ Scraping complete! Collected {len(scraper.jobs)} unique jobs")
    print(f"📊 Data saved to ml/data/jobs_data.json and ml/data/jobs_data.csv")