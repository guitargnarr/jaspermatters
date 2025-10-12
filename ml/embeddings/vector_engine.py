"""
Vector Embeddings Engine
Handles semantic search using Sentence Transformers and Pinecone
"""

import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import json
import os
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Structured embedding search result"""
    job_id: str
    title: str
    company: str
    score: float
    description: str
    requirements: List[str]
    salary_range: Optional[tuple]
    metadata: Dict[str, Any]


class VectorEngine:
    """Semantic search engine for job matching"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a lightweight but effective model"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.embeddings_cache = {}
        self.local_index = None  # For local testing without Pinecone
        self.use_local = True  # Start with local mode
        
        logger.info(f"Initialized VectorEngine with model: {model_name}")
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def initialize_pinecone(self, api_key: str, environment: str = "gcp-starter"):
        """Initialize Pinecone connection"""
        try:
            pc = Pinecone(api_key=api_key)
            
            index_name = "job-intelligence"
            
            # Create index if it doesn't exist
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created Pinecone index: {index_name}")
            
            self.index = pc.Index(index_name)
            self.use_local = False
            logger.info("Connected to Pinecone successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone: {e}. Using local mode.")
            self.use_local = True
    
    def create_job_embedding(self, job: Dict[str, Any]) -> np.ndarray:
        """Create embedding for a job posting"""
        # Combine relevant text fields for rich representation
        text_components = []
        
        # Title is most important
        if job.get('title'):
            text_components.append(f"Job Title: {job['title']}")
        
        # Company context
        if job.get('company'):
            text_components.append(f"Company: {job['company']}")
        
        # Description provides context
        if job.get('description'):
            desc = job['description'][:500]  # Limit length
            text_components.append(f"Description: {desc}")
        
        # Requirements are crucial for matching
        if job.get('requirements'):
            reqs = ', '.join(job['requirements'][:10])  # Top 10 requirements
            text_components.append(f"Requirements: {reqs}")
        
        # Seniority level
        if job.get('seniority_level'):
            text_components.append(f"Level: {job['seniority_level']}")
        
        # Combine all text
        combined_text = " | ".join(text_components)
        
        # Generate embedding
        embedding = self.model.encode(combined_text, convert_to_numpy=True)
        
        return embedding
    
    def create_resume_embedding(self, resume_text: str, target_role: Optional[str] = None) -> np.ndarray:
        """Create embedding for a resume/candidate profile"""
        if target_role:
            # Prepend target role for better matching
            resume_text = f"Target Role: {target_role} | {resume_text}"
        
        embedding = self.model.encode(resume_text, convert_to_numpy=True)
        return embedding
    
    def index_jobs(self, jobs_file: str):
        """Index all jobs from the scraped data"""
        logger.info(f"Loading jobs from {jobs_file}")
        
        # Load jobs
        with open(jobs_file, 'r') as f:
            jobs = json.load(f)
        
        logger.info(f"Indexing {len(jobs)} jobs...")
        
        if self.use_local:
            # Local mode: store embeddings in memory
            self.local_index = {
                'embeddings': [],
                'metadata': [],
                'ids': []
            }
            
            for i, job in enumerate(jobs):
                embedding = self.create_job_embedding(job)
                
                self.local_index['embeddings'].append(embedding)
                self.local_index['ids'].append(job.get('job_id', str(i)))
                self.local_index['metadata'].append({
                    'title': job.get('title', ''),
                    'company': job.get('company', ''),
                    'description': job.get('description', '')[:500],
                    'requirements': job.get('requirements', []),
                    'salary_min': job.get('salary_min'),
                    'salary_max': job.get('salary_max'),
                    'seniority_level': job.get('seniority_level', ''),
                    'remote': job.get('remote', False),
                    'source': job.get('source', ''),
                    'url': job.get('url', '')
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Indexed {i + 1}/{len(jobs)} jobs")
            
            # Convert to numpy array for efficient similarity computation
            self.local_index['embeddings'] = np.array(self.local_index['embeddings'])
            
            # Save local index
            import os
            index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'local_index.pkl')
            with open(index_path, 'wb') as f:
                pickle.dump(self.local_index, f)
            
            logger.info("Local indexing complete")
            
        else:
            # Pinecone mode
            batch_size = 100
            for i in range(0, len(jobs), batch_size):
                batch = jobs[i:i + batch_size]
                
                vectors = []
                for job in batch:
                    embedding = self.create_job_embedding(job)
                    
                    vectors.append({
                        'id': job.get('job_id', str(i)),
                        'values': embedding.tolist(),
                        'metadata': {
                            'title': job.get('title', ''),
                            'company': job.get('company', ''),
                            'description': job.get('description', '')[:500],
                            'requirements': str(job.get('requirements', [])),
                            'salary_min': job.get('salary_min', 0),
                            'salary_max': job.get('salary_max', 0),
                            'seniority_level': job.get('seniority_level', ''),
                            'remote': job.get('remote', False)
                        }
                    })
                
                self.index.upsert(vectors=vectors)
                logger.info(f"Indexed batch {i // batch_size + 1}")
            
            logger.info("Pinecone indexing complete")
    
    def search_jobs(self, 
                   query: str, 
                   top_k: int = 10,
                   filters: Optional[Dict[str, Any]] = None) -> List[EmbeddingResult]:
        """Search for jobs using semantic similarity"""
        
        # Create query embedding
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        if self.use_local:
            # Local similarity search
            if self.local_index is None:
                # Try to load saved index
                try:
                    import os
                    index_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'local_index.pkl')
                    with open(index_path, 'rb') as f:
                        self.local_index = pickle.load(f)
                except:
                    logger.error("No local index found. Please run index_jobs first.")
                    return []
            
            # Compute cosine similarities
            similarities = cosine_similarity(
                [query_embedding], 
                self.local_index['embeddings']
            )[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                metadata = self.local_index['metadata'][idx]
                
                # Apply filters if provided
                if filters:
                    skip = False
                    if 'remote' in filters and metadata.get('remote') != filters['remote']:
                        skip = True
                    if 'seniority_level' in filters and metadata.get('seniority_level') != filters['seniority_level']:
                        skip = True
                    if 'min_salary' in filters:
                        max_sal = metadata.get('salary_max', 0)
                        if max_sal and max_sal < filters['min_salary']:
                            skip = True
                    
                    if skip:
                        continue
                
                results.append(EmbeddingResult(
                    job_id=self.local_index['ids'][idx],
                    title=metadata['title'],
                    company=metadata['company'],
                    score=float(similarities[idx]),
                    description=metadata['description'],
                    requirements=metadata['requirements'],
                    salary_range=(metadata.get('salary_min'), metadata.get('salary_max')),
                    metadata=metadata
                ))
            
            return results[:top_k]
            
        else:
            # Pinecone search
            filter_dict = {}
            if filters:
                if 'remote' in filters:
                    filter_dict['remote'] = filters['remote']
                if 'seniority_level' in filters:
                    filter_dict['seniority_level'] = filters['seniority_level']
                # Note: Pinecone doesn't support range queries in free tier
            
            response = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None
            )
            
            results = []
            for match in response['matches']:
                metadata = match['metadata']
                results.append(EmbeddingResult(
                    job_id=match['id'],
                    title=metadata['title'],
                    company=metadata['company'],
                    score=match['score'],
                    description=metadata['description'],
                    requirements=eval(metadata.get('requirements', '[]')),
                    salary_range=(metadata.get('salary_min'), metadata.get('salary_max')),
                    metadata=metadata
                ))
            
            return results
    
    def match_resume_to_jobs(self, 
                           resume_text: str,
                           top_k: int = 20) -> List[EmbeddingResult]:
        """Match a resume to most relevant jobs"""
        
        # Extract key skills from resume for better matching
        skills_keywords = [
            'Python', 'TensorFlow', 'PyTorch', 'Machine Learning',
            'Deep Learning', 'SQL', 'Docker', 'Kubernetes', 'AWS'
        ]
        
        found_skills = []
        for skill in skills_keywords:
            if skill.lower() in resume_text.lower():
                found_skills.append(skill)
        
        # Create enhanced query
        if found_skills:
            query = f"Skills: {', '.join(found_skills)} | {resume_text[:500]}"
        else:
            query = resume_text[:1000]
        
        return self.search_jobs(query, top_k=top_k)
    
    def find_similar_jobs(self, job_id: str, top_k: int = 5) -> List[EmbeddingResult]:
        """Find jobs similar to a given job"""
        
        if self.use_local:
            # Find the job embedding
            try:
                idx = self.local_index['ids'].index(job_id)
                job_embedding = self.local_index['embeddings'][idx]
                
                # Compute similarities
                similarities = cosine_similarity(
                    [job_embedding],
                    self.local_index['embeddings']
                )[0]
                
                # Get top-k+1 (excluding the job itself)
                top_indices = np.argsort(similarities)[-(top_k + 1):-1][::-1]
                
                results = []
                for idx in top_indices:
                    metadata = self.local_index['metadata'][idx]
                    results.append(EmbeddingResult(
                        job_id=self.local_index['ids'][idx],
                        title=metadata['title'],
                        company=metadata['company'],
                        score=float(similarities[idx]),
                        description=metadata['description'],
                        requirements=metadata['requirements'],
                        salary_range=(metadata.get('salary_min'), metadata.get('salary_max')),
                        metadata=metadata
                    ))
                
                return results
                
            except ValueError:
                logger.error(f"Job ID {job_id} not found in index")
                return []
        else:
            # Pinecone mode - would need to fetch the job first
            # For now, return empty
            return []
    
    def skill_gap_analysis(self, resume_text: str, job_ids: List[str]) -> Dict[str, Any]:
        """Analyze skill gaps between resume and target jobs"""
        
        # Common skills to check
        all_skills = [
            'Python', 'Java', 'JavaScript', 'SQL', 'R', 'Scala',
            'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras',
            'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure',
            'Git', 'CI/CD', 'Agile', 'REST API', 'GraphQL',
            'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
            'Statistics', 'Mathematics', 'Data Analysis'
        ]
        
        # Extract skills from resume
        resume_skills = set()
        for skill in all_skills:
            if skill.lower() in resume_text.lower():
                resume_skills.add(skill)
        
        # Extract skills from target jobs
        job_skills = set()
        if self.use_local and self.local_index:
            for job_id in job_ids:
                try:
                    idx = self.local_index['ids'].index(job_id)
                    requirements = self.local_index['metadata'][idx]['requirements']
                    for skill in all_skills:
                        if skill in str(requirements):
                            job_skills.add(skill)
                except:
                    continue
        
        # Calculate gaps
        missing_skills = job_skills - resume_skills
        matching_skills = resume_skills & job_skills
        
        # Prioritize missing skills by frequency
        skill_frequency = {}
        for skill in missing_skills:
            count = 0
            if self.use_local and self.local_index:
                for metadata in self.local_index['metadata']:
                    if skill in str(metadata.get('requirements', [])):
                        count += 1
            skill_frequency[skill] = count
        
        # Sort by frequency
        priority_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'priority_skills': priority_skills[:10],  # Top 10 most important to learn
            'match_percentage': len(matching_skills) / len(job_skills) * 100 if job_skills else 0
        }


if __name__ == "__main__":
    # Test the vector engine
    import os
    engine = VectorEngine()
    
    # Get the correct path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    jobs_file = os.path.join(base_dir, 'ml', 'data', 'jobs_data.json')
    
    # Index the jobs we scraped
    engine.index_jobs(jobs_file)
    
    # Test search
    results = engine.search_jobs("machine learning engineer with Python experience", top_k=5)
    
    print("\n🔍 Search Results for 'machine learning engineer with Python':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title} at {result.company}")
        print(f"   Score: {result.score:.3f}")
        print(f"   Salary: ${result.salary_range[0]:,.0f} - ${result.salary_range[1]:,.0f}" 
              if result.salary_range[0] else "   Salary: Not specified")
        print(f"   Requirements: {', '.join(result.requirements[:5])}")