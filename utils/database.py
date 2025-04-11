import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json
import io
import pickle

# Create SQLAlchemy engine and session with a local SQLite database
# Instead of using an environment variable, use a local SQLite database
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./fire_app.db')

# Configure connection arguments based on database type
connect_args = {}
if DATABASE_URL.startswith('sqlite'):
    # SQLite specific arguments
    connect_args = {'check_same_thread': False}
else:
    # PostgreSQL/MySQL arguments
    connect_args = {
        "connect_timeout": 10,
        "application_name": "FIRE app"
    }

# Add connection pooling and retry parameters
engine = create_engine(
    DATABASE_URL, 
    pool_pre_ping=True,  # Test connections before using them
    pool_recycle=3600,   # Recycle connections after 1 hour
    pool_timeout=30,     # Timeout for getting a connection from pool
    pool_size=5,         # Small pool size for Replit environment
    max_overflow=10,     # Maximum number of overflow connections
    connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define models
class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String, index=True)  # csv, xlsx, etc.
    upload_date = Column(DateTime, default=datetime.now)
    file_data = Column(LargeBinary)  # Store the file data
    
class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(Text)
    query_date = Column(DateTime, default=datetime.now)
    result_data = Column(Text)  # JSON string of results
    file_id = Column(Integer)  # Reference to the file queried
    visualization_type = Column(String)

class VisualizationHistory(Base):
    __tablename__ = "visualization_history"

    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer)  # Reference to the query
    viz_type = Column(String)
    result_data = Column(Text)  # JSON string of data used for visualization
    creation_date = Column(DateTime, default=datetime.now)

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String, index=True)  # e.g., "default_viz", "theme", etc.
    setting_value = Column(Text)
    updated_date = Column(DateTime, default=datetime.now)

# Create tables
Base.metadata.create_all(bind=engine)

# Database Functions
def get_db_session():
    """Get a database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

def save_file_to_db(file_object, filename):
    """Save uploaded file to the database with retry logic and fallback to local storage"""
    import time
    import os
    from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
    
    file_type = filename.split('.')[-1].lower()
    
    # Store the binary content
    file_object.seek(0)
    file_data = file_object.read()
    
    # Retry configuration - 3 retry attempts with exponential backoff
    max_retries = 3
    retry_count = 0
    retry_delay = 1  # Start with 1 second delay
    
    while retry_count < max_retries:
        db = get_db_session()
        try:
            db_file = UploadedFile(
                filename=filename,
                file_type=file_type, 
                file_data=file_data
            )
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            return db_file.id  # Success, return the ID
            
        except (OperationalError, IntegrityError) as e:
            # Connection or data integrity issues
            db.rollback()
            retry_count += 1
            if retry_count >= max_retries:
                # If all retries failed, resort to local file storage
                # Store file locally as a fallback and generate a fake ID
                local_storage_path = os.path.join("tmp", "fallback_files")
                os.makedirs(local_storage_path, exist_ok=True)
                
                # Create a unique ID for the local file
                import hashlib
                import uuid
                unique_id = str(uuid.uuid4())
                file_hash = hashlib.md5(file_data).hexdigest()[:8]  # First 8 chars of MD5 hash
                local_id = f"local_{file_hash}_{unique_id[:8]}"
                
                # Save the file locally
                local_file_path = os.path.join(local_storage_path, f"{local_id}.{file_type}")
                with open(local_file_path, 'wb') as f:
                    f.write(file_data)
                
                # Register the local file in a "registry" for later lookups
                registry_path = os.path.join(local_storage_path, "file_registry.json")
                import json
                registry = {}
                if os.path.exists(registry_path):
                    try:
                        with open(registry_path, 'r') as f:
                            registry = json.load(f)
                    except:
                        pass  # If registry is corrupted, start fresh
                
                registry[local_id] = {
                    'filename': filename,
                    'file_type': file_type,
                    'local_path': local_file_path,
                    'upload_date': str(datetime.now())
                }
                
                with open(registry_path, 'w') as f:
                    json.dump(registry, f)
                    
                return local_id
                
            # Wait before retrying with exponential backoff
            time.sleep(retry_delay)
            retry_delay *= 2  # Double the delay for next retry
            
        except SQLAlchemyError as e:
            # Other database errors - try local file storage
            db.rollback()
            # Use the same local storage code as above, but as a separate function in a real app
            # This is duplicated here for simplicity
            local_storage_path = os.path.join("tmp", "fallback_files")
            os.makedirs(local_storage_path, exist_ok=True)
            
            # Create a unique ID for the local file
            import hashlib
            import uuid
            unique_id = str(uuid.uuid4())
            file_hash = hashlib.md5(file_data).hexdigest()[:8]
            local_id = f"local_{file_hash}_{unique_id[:8]}"
            
            # Save the file locally
            local_file_path = os.path.join(local_storage_path, f"{local_id}.{file_type}")
            with open(local_file_path, 'wb') as f:
                f.write(file_data)
            
            # Register the local file
            registry_path = os.path.join(local_storage_path, "file_registry.json")
            import json
            registry = {}
            if os.path.exists(registry_path):
                try:
                    with open(registry_path, 'r') as f:
                        registry = json.load(f)
                except:
                    pass  # If registry is corrupted, start fresh
            
            registry[local_id] = {
                'filename': filename,
                'file_type': file_type,
                'local_path': local_file_path,
                'upload_date': str(datetime.now())
            }
            
            with open(registry_path, 'w') as f:
                json.dump(registry, f)
                
            return local_id
            
        finally:
            db.close()

def get_file_from_db(file_id):
    """Retrieve a file from the database or local storage by its ID"""
    import os
    
    # Check if this is a local fallback ID
    if isinstance(file_id, str) and file_id.startswith('local_'):
        # Load from local fallback storage
        local_storage_path = os.path.join("tmp", "fallback_files")
        registry_path = os.path.join(local_storage_path, "file_registry.json")
        
        # Check if registry exists
        if not os.path.exists(registry_path):
            return None
            
        # Load registry
        try:
            import json
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except:
            return None
            
        # Check if file_id exists in registry
        if file_id not in registry:
            return None
            
        file_info = registry[file_id]
        file_path = file_info['local_path']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None
            
        # Load file data
        with open(file_path, 'rb') as f:
            file_data = f.read()
            
        # Return file info in the same format as database files
        return {
            'id': file_id,
            'filename': file_info['filename'],
            'file_type': file_info['file_type'],
            'upload_date': file_info['upload_date'],
            'file_data': file_data
        }
    
    # If not a local ID, load from database
    db = get_db_session()
    try:
        file = db.query(UploadedFile).filter(UploadedFile.id == file_id).first()
        if file:
            return {
                'id': file.id,
                'filename': file.filename,
                'file_type': file.file_type,
                'upload_date': file.upload_date,
                'file_data': file.file_data
            }
        return None
    finally:
        db.close()

def get_all_files():
    """Get a list of all uploaded files including local fallback files"""
    import os
    
    result_files = []
    
    # Get DB files
    db = get_db_session()
    try:
        db_files = db.query(UploadedFile).all()
        result_files = [{'id': f.id, 'filename': f.filename, 'file_type': f.file_type, 'upload_date': f.upload_date} 
                      for f in db_files]
    except:
        # If DB query fails, continue with local files only
        pass
    finally:
        db.close()
    
    # Get local files from registry
    local_storage_path = os.path.join("tmp", "fallback_files")
    registry_path = os.path.join(local_storage_path, "file_registry.json")
    
    if os.path.exists(registry_path):
        try:
            import json
            with open(registry_path, 'r') as f:
                registry = json.load(f)
                
            # Add local files to the result
            for file_id, file_info in registry.items():
                try:
                    # Check if file exists
                    if os.path.exists(file_info['local_path']):
                        result_files.append({
                            'id': file_id,
                            'filename': file_info['filename'],
                            'file_type': file_info['file_type'],
                            'upload_date': file_info['upload_date'],
                            'is_local': True  # Flag to indicate this is a local file
                        })
                except:
                    continue  # Skip problematic entries
        except:
            pass  # If registry is corrupted, return DB files only
    
    return result_files

def load_dataframe_from_file(file_id):
    """Load a file from DB or local fallback storage as a pandas DataFrame"""
    import os
    
    # Check if this is a local fallback ID
    if isinstance(file_id, str) and file_id.startswith('local_'):
        # Load from local fallback storage
        local_storage_path = os.path.join("tmp", "fallback_files")
        registry_path = os.path.join(local_storage_path, "file_registry.json")
        
        # Check if registry exists
        if not os.path.exists(registry_path):
            return None
            
        # Load registry
        try:
            import json
            with open(registry_path, 'r') as f:
                registry = json.load(f)
        except:
            return None
            
        # Check if file_id exists in registry
        if file_id not in registry:
            return None
            
        file_info = registry[file_id]
        file_path = file_info['local_path']
        file_type = file_info['file_type']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return None
            
        # Load based on file type
        if file_type == 'csv':
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin1')
                except:
                    df = pd.read_csv(file_path, encoding='cp1252')
            except:
                # Try different delimiters
                df = pd.read_csv(file_path, sep=';')
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
            
        return df
    
    # If not a local ID, try loading from database
    file_info = get_file_from_db(file_id)
    if not file_info:
        return None
    
    file_data = file_info['file_data']
    file_type = file_info['file_type']
    
    # Create file-like object from the binary data
    file_like = io.BytesIO(file_data)
    
    # Load based on file type
    if file_type == 'csv':
        # Try different encodings
        try:
            df = pd.read_csv(file_like, encoding='utf-8')
        except UnicodeDecodeError:
            file_like.seek(0)
            try:
                df = pd.read_csv(file_like, encoding='latin1')
            except:
                file_like.seek(0)
                df = pd.read_csv(file_like, encoding='cp1252')
        except:
            # Try different delimiters
            file_like.seek(0)
            df = pd.read_csv(file_like, sep=';')
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file_like)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return df

def save_query_to_db(query_text, result_df, file_id, viz_type):
    """Save a query and its results to the database"""
    # Convert DataFrame to JSON string
    result_json = result_df.to_json(orient='records')
    
    db = get_db_session()
    try:
        db_query = QueryHistory(
            query_text=query_text,
            result_data=result_json,
            file_id=file_id,
            visualization_type=viz_type
        )
        db.add(db_query)
        db.commit()
        db.refresh(db_query)
        return db_query.id
    finally:
        db.close()

def get_query_history(limit=10):
    """Get recent query history"""
    db = get_db_session()
    try:
        queries = db.query(QueryHistory).order_by(QueryHistory.query_date.desc()).limit(limit).all()
        return [{
            'id': q.id,
            'query_text': q.query_text,
            'query_date': q.query_date,
            'file_id': q.file_id,
            'visualization_type': q.visualization_type,
            'result_data': json.loads(q.result_data) if q.result_data else {}
        } for q in queries]
    finally:
        db.close()

def save_visualization_to_db(query_id, viz_type, result_df):
    """Save a visualization to the database"""
    # Convert DataFrame to JSON string
    result_json = result_df.to_json(orient='records')
    
    db = get_db_session()
    try:
        db_viz = VisualizationHistory(
            query_id=query_id,
            viz_type=viz_type,
            result_data=result_json
        )
        db.add(db_viz)
        db.commit()
        db.refresh(db_viz)
        return db_viz.id
    finally:
        db.close()

def get_visualization_history(limit=5):
    """Get recent visualization history"""
    db = get_db_session()
    try:
        visualizations = db.query(VisualizationHistory).order_by(VisualizationHistory.creation_date.desc()).limit(limit).all()
        query_ids = [v.query_id for v in visualizations]
        
        # Get the associated queries for context
        queries = {}
        if query_ids:
            query_objs = db.query(QueryHistory).filter(QueryHistory.id.in_(query_ids)).all()
            queries = {q.id: q.query_text for q in query_objs}
            
        return [{
            'id': v.id,
            'query_id': v.query_id,
            'query_text': queries.get(v.query_id, "Unknown query"),
            'viz_type': v.viz_type,
            'creation_date': v.creation_date,
            'result_data': json.loads(v.result_data) if v.result_data else {}
        } for v in visualizations]
    finally:
        db.close()

def save_user_preference(key, value):
    """Save a user preference setting"""
    db = get_db_session()
    try:
        # Check if key already exists
        existing = db.query(UserPreference).filter(UserPreference.setting_key == key).first()
        
        if existing:
            existing.setting_value = value
            existing.updated_date = datetime.now()
        else:
            new_pref = UserPreference(
                setting_key=key,
                setting_value=value
            )
            db.add(new_pref)
            
        db.commit()
        return True
    finally:
        db.close()

def get_user_preference(key, default=None):
    """Get a user preference setting"""
    db = get_db_session()
    try:
        pref = db.query(UserPreference).filter(UserPreference.setting_key == key).first()
        return pref.setting_value if pref else default
    finally:
        db.close()