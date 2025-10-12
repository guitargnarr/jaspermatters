"""
Job Market Clustering and Segmentation
Uses K-means and DBSCAN for market analysis
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobMarketClusterer:
    """Cluster jobs to identify market segments"""
    
    def __init__(self):
        self.kmeans_model = None
        self.dbscan_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.cluster_labels = None
        self.cluster_centers = None
        
    def extract_clustering_features(self, jobs_df: pd.DataFrame) -> np.ndarray:
        """Extract features for clustering"""
        features = []
        
        # Salary features (normalized)
        salary_mid = (jobs_df['salary_min'].fillna(50000) + 
                     jobs_df['salary_max'].fillna(150000)) / 2
        features.append(salary_mid / 100000)  # Scale to 0-2 range typically
        
        # Seniority level encoding
        seniority_map = {
            'Junior': 0, 'Mid-level': 1, 'Senior': 2, 
            'Lead': 3, 'Principal': 4, 'Management': 3
        }
        seniority_scores = jobs_df['seniority_level'].map(
            lambda x: seniority_map.get(x, 1)
        )
        features.append(seniority_scores)
        
        # Remote work indicator
        features.append(jobs_df['remote'].astype(int))
        
        # Number of requirements (complexity indicator)
        num_requirements = jobs_df['requirements'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        features.append(num_requirements / 10)  # Normalize
        
        # Key technology indicators
        tech_categories = {
            'ml_deep': ['TensorFlow', 'PyTorch', 'Keras', 'Deep Learning'],
            'cloud': ['AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes'],
            'data_eng': ['Spark', 'Hadoop', 'Kafka', 'Airflow', 'ETL'],
            'web_dev': ['React', 'Node.js', 'JavaScript', 'REST API', 'GraphQL'],
            'traditional_ml': ['scikit-learn', 'Statistics', 'R', 'SAS']
        }
        
        for category, keywords in tech_categories.items():
            category_score = jobs_df['requirements'].apply(
                lambda reqs: sum(1 for kw in keywords 
                               if isinstance(reqs, list) and kw in str(reqs)) / len(keywords)
            )
            features.append(category_score)
        
        # Description length (normalized)
        desc_length = jobs_df['description'].apply(
            lambda x: len(str(x)) if x else 0
        ) / 1000
        features.append(desc_length)
        
        # Convert to numpy array
        feature_matrix = np.column_stack(features)
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features for clustering")
        return feature_matrix
    
    def find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method and silhouette score"""
        inertias = []
        silhouette_scores = []
        
        for k in range(2, min(max_k + 1, len(X))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
        
        # Find elbow point (simplified - in production use more sophisticated methods)
        # Look for the point where the rate of decrease sharply changes
        if len(inertias) > 2:
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            elbow = np.argmax(diffs2) + 2  # +2 because we start from k=2
        else:
            elbow = 3
        
        # Also consider silhouette score
        best_silhouette = np.argmax(silhouette_scores) + 2
        
        # Balance between elbow and silhouette
        optimal_k = int((elbow + best_silhouette) / 2)
        
        logger.info(f"Optimal clusters - Elbow: {elbow}, Silhouette: {best_silhouette}, Selected: {optimal_k}")
        
        # Plot analysis
        if len(inertias) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(range(2, len(inertias) + 2), inertias, 'bo-')
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Inertia')
            ax1.set_title('Elbow Method')
            ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
            ax1.legend()
            ax1.grid(True)
            
            ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'go-')
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Silhouette Score')
            ax2.set_title('Silhouette Analysis')
            ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig('ml/models/cluster_analysis.png')
            logger.info("Cluster analysis saved to ml/models/cluster_analysis.png")
        
        return optimal_k
    
    def perform_kmeans_clustering(self, X: np.ndarray, n_clusters: Optional[int] = None):
        """Perform K-means clustering"""
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,
            max_iter=300
        )
        
        self.cluster_labels = self.kmeans_model.fit_predict(X_scaled)
        self.cluster_centers = self.scaler.inverse_transform(self.kmeans_model.cluster_centers_)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, self.cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, self.cluster_labels)
        
        logger.info(f"K-means clustering complete: {n_clusters} clusters")
        logger.info(f"Silhouette Score: {silhouette:.3f}")
        logger.info(f"Calinski-Harabasz Score: {calinski:.3f}")
        
        return self.cluster_labels
    
    def perform_dbscan_clustering(self, X: np.ndarray, eps: float = 0.5, min_samples: int = 3):
        """Perform DBSCAN for outlier detection"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.dbscan_model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean'
        )
        
        dbscan_labels = self.dbscan_model.fit_predict(X_scaled)
        
        n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        n_outliers = list(dbscan_labels).count(-1)
        
        logger.info(f"DBSCAN clustering complete: {n_clusters} clusters, {n_outliers} outliers")
        
        return dbscan_labels
    
    def visualize_clusters(self, X: np.ndarray, labels: np.ndarray, title: str = "Job Market Clusters"):
        """Visualize clusters using PCA"""
        # Scale and reduce dimensions
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Get unique labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Outliers in DBSCAN
                label_name = 'Outliers'
                marker = 'x'
            else:
                label_name = f'Cluster {label}'
                marker = 'o'
            
            mask = labels == label
            plt.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[color],
                label=label_name,
                marker=marker,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
        
        # Add cluster centers if K-means
        if self.cluster_centers is not None:
            centers_scaled = self.scaler.transform(self.cluster_centers)
            centers_pca = self.pca.transform(centers_scaled)
            plt.scatter(
                centers_pca[:, 0],
                centers_pca[:, 1],
                c='red',
                marker='*',
                s=500,
                edgecolors='black',
                linewidth=2,
                label='Centroids'
            )
        
        plt.xlabel(f'First Principal Component ({self.pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Second Principal Component ({self.pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('ml/models/cluster_visualization.png', dpi=150)
        logger.info("Cluster visualization saved to ml/models/cluster_visualization.png")
    
    def analyze_clusters(self, jobs_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Analyze characteristics of each cluster"""
        jobs_df['cluster'] = labels
        
        cluster_analysis = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                continue  # Skip outliers
            
            cluster_jobs = jobs_df[jobs_df['cluster'] == cluster_id]
            
            # Calculate cluster statistics
            avg_salary = (cluster_jobs['salary_min'].fillna(50000) + 
                         cluster_jobs['salary_max'].fillna(150000)).mean() / 2
            
            # Most common seniority level
            seniority_counts = cluster_jobs['seniority_level'].value_counts()
            dominant_seniority = seniority_counts.index[0] if len(seniority_counts) > 0 else 'Unknown'
            
            # Remote percentage
            remote_pct = cluster_jobs['remote'].mean() * 100
            
            # Most common requirements
            all_requirements = []
            for reqs in cluster_jobs['requirements']:
                if isinstance(reqs, list):
                    all_requirements.extend(reqs)
            
            if all_requirements:
                req_counts = pd.Series(all_requirements).value_counts()
                top_skills = req_counts.head(5).index.tolist()
            else:
                top_skills = []
            
            cluster_analysis[f'Cluster_{cluster_id}'] = {
                'size': len(cluster_jobs),
                'avg_salary': avg_salary,
                'dominant_seniority': dominant_seniority,
                'remote_percentage': remote_pct,
                'top_skills': top_skills,
                'companies': cluster_jobs['company'].value_counts().head(3).index.tolist()
            }
        
        return cluster_analysis
    
    def get_cluster_recommendations(self, cluster_id: int) -> Dict:
        """Get recommendations based on cluster characteristics"""
        if self.cluster_labels is None:
            return {"error": "No clustering performed yet"}
        
        recommendations = {
            'cluster_id': cluster_id,
            'career_advice': [],
            'skill_recommendations': [],
            'job_search_tips': []
        }
        
        # This would be enhanced with actual cluster analysis
        if cluster_id == 0:
            recommendations['career_advice'] = [
                "Entry-level cluster: Focus on building foundational skills",
                "Consider internships and junior positions",
                "Build a strong portfolio of projects"
            ]
        elif cluster_id == 1:
            recommendations['career_advice'] = [
                "Mid-level cluster: Time to specialize",
                "Consider leadership opportunities",
                "Expand your technical depth"
            ]
        else:
            recommendations['career_advice'] = [
                "Senior cluster: Focus on system design and architecture",
                "Develop mentoring and leadership skills",
                "Consider consulting or technical leadership roles"
            ]
        
        return recommendations
    
    def save_models(self):
        """Save clustering models"""
        models = {
            'kmeans': self.kmeans_model,
            'dbscan': self.dbscan_model,
            'scaler': self.scaler,
            'pca': self.pca,
            'cluster_labels': self.cluster_labels
        }
        
        with open('ml/models/clustering_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        logger.info("Clustering models saved to ml/models/clustering_models.pkl")


if __name__ == "__main__":
    import os
    
    # Load job data
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    jobs_file = os.path.join(base_dir, 'ml', 'data', 'jobs_data.json')
    
    with open(jobs_file, 'r') as f:
        jobs_data = json.load(f)
    
    jobs_df = pd.DataFrame(jobs_data)
    
    # Initialize clusterer
    clusterer = JobMarketClusterer()
    
    # Extract features
    X = clusterer.extract_clustering_features(jobs_df)
    
    # Perform K-means clustering
    print("\n🎯 Performing K-means Clustering...")
    kmeans_labels = clusterer.perform_kmeans_clustering(X, n_clusters=3)
    
    # Visualize clusters
    clusterer.visualize_clusters(X, kmeans_labels, "Job Market Segmentation (K-means)")
    
    # Analyze clusters
    cluster_analysis = clusterer.analyze_clusters(jobs_df, kmeans_labels)
    
    print("\n📊 Cluster Analysis:")
    print("-" * 60)
    for cluster_name, stats in cluster_analysis.items():
        print(f"\n{cluster_name}:")
        print(f"  Size: {stats['size']} jobs")
        print(f"  Avg Salary: ${stats['avg_salary']:,.0f}")
        print(f"  Dominant Level: {stats['dominant_seniority']}")
        print(f"  Remote: {stats['remote_percentage']:.1f}%")
        print(f"  Top Skills: {', '.join(stats['top_skills'][:3])}")
    
    # Perform DBSCAN for outlier detection
    print("\n🔍 Performing DBSCAN Outlier Detection...")
    dbscan_labels = clusterer.perform_dbscan_clustering(X, eps=1.5, min_samples=2)
    
    # Save models
    clusterer.save_models()
    
    print("\n✅ Clustering analysis complete!")