"""
Session state management for maintaining training states and results
"""

import streamlit as st
import pickle
import os
from datetime import datetime
from config import Config


class SessionManager:
    """Manages session state and persistent storage of training results"""

    def __init__(self):
        self.session_file = "wildlife_session.pkl"
        self.initialize_session_state()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        # Training status flags
        for key in Config.SESSION_KEYS.values():
            if key not in st.session_state:
                st.session_state[key] = False if key.endswith(('_trained', '_loaded', '_completed')) else None

        # Additional initialization
        if 'training_history' not in st.session_state:
            st.session_state.training_history = []

        if 'last_training_time' not in st.session_state:
            st.session_state.last_training_time = None

        if 'current_dataset_info' not in st.session_state:
            st.session_state.current_dataset_info = {}

    def save_session(self):
        """Save current session state to file"""
        try:
            session_data = {
                'timestamp': datetime.now(),
                'training_states': {
                    key: st.session_state.get(key, False)
                    for key in Config.SESSION_KEYS.values()
                    if key.endswith(('_trained', '_loaded', '_completed'))
                },
                'results_available': {
                    'ml_results': st.session_state.get('ml_results') is not None,
                    'cnn_results': st.session_state.get('cnn_results') is not None,
                    'resnet_results': st.session_state.get('resnet_results') is not None,
                    'ensemble_results': st.session_state.get('ensemble_results') is not None,
                    'eda_results': st.session_state.get('eda_results') is not None
                },
                'dataset_info': st.session_state.get('current_dataset_info', {}),
                'training_history': st.session_state.get('training_history', [])
            }

            with open(self.session_file, 'wb') as f:
                pickle.dump(session_data, f)

            return True
        except Exception as e:
            st.error(f"Failed to save session: {str(e)}")
            return False

    def load_session(self):
        """Load session state from file"""
        if not os.path.exists(self.session_file):
            return False

        try:
            with open(self.session_file, 'rb') as f:
                session_data = pickle.load(f)

            # Restore training states
            for key, value in session_data.get('training_states', {}).items():
                st.session_state[key] = value

            # Restore other data
            st.session_state.current_dataset_info = session_data.get('dataset_info', {})
            st.session_state.training_history = session_data.get('training_history', [])
            st.session_state.last_training_time = session_data.get('timestamp')

            return True
        except Exception as e:
            st.error(f"Failed to load session: {str(e)}")
            return False

    def clear_session(self):
        """Clear all session data"""
        for key in Config.SESSION_KEYS.values():
            if key in st.session_state:
                if key.endswith(('_trained', '_loaded', '_completed')):
                    st.session_state[key] = False
                else:
                    st.session_state[key] = None

        st.session_state.training_history = []
        st.session_state.last_training_time = None
        st.session_state.current_dataset_info = {}

        # Remove session file
        if os.path.exists(self.session_file):
            os.remove(self.session_file)

    def get_training_status(self):
        """Get comprehensive training status"""
        return {
            'EDA Completed': st.session_state.get('eda_completed', False),
            'Data Loaded': st.session_state.get('data_loaded', False),
            'ML Models Trained': st.session_state.get('ml_models_trained', False),
            'CNN Trained': st.session_state.get('cnn_trained', False),
            'ResNet Trained': st.session_state.get('resnet_trained', False),
            'Ensemble Trained': st.session_state.get('ensemble_trained', False)
        }

    def add_training_event(self, event_type, model_name, accuracy=None, duration=None):
        """Add a training event to history"""
        event = {
            'timestamp': datetime.now(),
            'event_type': event_type,
            'model_name': model_name,
            'accuracy': accuracy,
            'duration': duration
        }

        if 'training_history' not in st.session_state:
            st.session_state.training_history = []

        st.session_state.training_history.append(event)
        self.save_session()

    def get_training_history(self):
        """Get training history"""
        return st.session_state.get('training_history', [])

    def get_session_summary(self):
        """Get a summary of current session"""
        status = self.get_training_status()
        completed_tasks = sum(1 for task_completed in status.values() if task_completed)
        total_tasks = len(status)

        return {
            'completion_percentage': (completed_tasks / total_tasks) * 100,
            'completed_tasks': completed_tasks,
            'total_tasks': total_tasks,
            'last_activity': st.session_state.get('last_training_time'),
            'status': status
        }