import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import time
import plotly.express as px
import numpy as np

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# Configure page
st.set_page_config(
    page_title="Salary Benchmark", 
    page_icon="💼", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem 10rem;
    }
    .st-emotion-cache-1v0mbdj {
        border-radius: 15px;
    }
    .highlight {
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        margin-bottom: 20px;
    }
    .prediction-high {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
    }
    .prediction-low {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    .confidence-text, .description-text {
        color: #000000 !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=150)
with col2:
    st.title("Salary Benchmark")
    st.markdown("Predict your salary based on your experience")

# Sidebar with enhanced UI
with st.sidebar:
    st.header("📋 Employee Details")
    
    with st.expander("**Demographics**", expanded=True):
        age = st.slider("Age", 18, 65, 30, help="Employee's current age")
        education = st.selectbox("Education Level", [
            "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
        ], index=0, help="Highest education level achieved")
        
    with st.expander("**Employment Details**", expanded=True):
        occupation = st.selectbox("Job Role", [
            "Tech-support", "Craft-repair", "Other-service", "Sales",
            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
            "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
            "Protective-serv", "Armed-Forces"
        ], index=4, help="Current job role")
        hours_per_week = st.slider("Hours per week", 1, 80, 40, 
                                 help="Typical weekly working hours")
        experience = st.slider("Years of Experience", 0, 40, 5, 
                             help="Total years of professional experience")
    
    st.markdown("---")
    st.markdown("**Need help?**")
    st.info("""
    Fill in the employee details and click the **Predict** button 
    to see the salary classification.
    """)

# Input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

# Main content
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction", "📊 Insights"])

with tab1:
    st.markdown("### Input Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        # Visualize key features
        st.markdown("#### Key Features Visualization")
        fig = px.bar(
            x=['Age', 'Education', 'Experience', 'Hours/Week'],
            y=[age, 1, experience, hours_per_week],  # Using 1 for categorical for visualization
            labels={'x': 'Feature', 'y': 'Value'},
            color=['Age', 'Education', 'Experience', 'Hours/Week'],
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Prediction button with enhanced UI
        if st.button("🔮 Predict Salary Class", use_container_width=True):
            with st.spinner("Analyzing employee data..."):
                # Add progress bar for better UX
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0]
                
                # Display prediction with confidence
                if prediction == '>50K':
                    confidence = prediction_proba[1] * 100
                    st.markdown(f"""
                    <div class="highlight prediction-high">
                        <h2 style='color:#155724'>💰 High Earner Prediction</h2>
                        <p style='font-size:24px;color:#155724'>Salary: <strong>>50K</strong></p>
                        <p class="confidence-text">Confidence: <strong>{confidence:.1f}%</strong></p>
                        <p class="description-text">This employee is likely in the higher salary bracket based on their profile.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show what's driving the prediction
                    st.markdown("#### 🚀 What's driving this prediction?")
                    st.info("""
                    - Higher education levels (Master's/PhD)
                    - Managerial or professional occupations
                    - More years of experience
                    - Higher weekly working hours
                    """)
                else:
                    confidence = prediction_proba[0] * 100
                    st.markdown(f"""
                    <div class="highlight prediction-low">
                        <h2 style='color:#721c24'>💸 Standard Salary Prediction</h2>
                        <p style='font-size:24px;color:#721c24'>Salary: <strong>≤50K</strong></p>
                        <p class="confidence-text">Confidence: <strong>{confidence:.1f}%</strong></p>
                        <p class="description-text">This employee is likely in the standard salary bracket based on their profile.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show what might increase salary potential
                    st.markdown("#### 📈 How to improve salary potential?")
                    st.info("""
                    - Pursue higher education (Master's/PhD)
                    - Transition to managerial/professional roles
                    - Gain more work experience
                    - Consider working more hours per week
                    """)

with tab2:
    st.markdown("## 📂 Batch Prediction")
    st.info("Upload a CSV file containing multiple employee records to get predictions for all of them at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", 
                                   help="File should contain columns: age, education, occupation, hours-per-week")
    
    if uploaded_file is not None:
        batch_data = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully uploaded {len(batch_data)} records!")
        
        with st.expander("View Uploaded Data", expanded=True):
            st.dataframe(batch_data, use_container_width=True)
        
        # Add 'experience' column if missing
        if 'experience' not in batch_data.columns and 'age' in batch_data.columns:
            batch_data['experience'] = batch_data['age'] - 18
        
        # Ensure required columns are present
        expected_cols = ['age', 'education', 'occupation', 'hours-per-week', 'experience']
        missing_cols = set(expected_cols) - set(batch_data.columns)
        
        if missing_cols:
            st.error(f"❗ Missing required columns: {', '.join(missing_cols)}")
        else:
            if st.button("🚀 Run Batch Prediction", use_container_width=True):
                with st.spinner("Processing batch predictions..."):
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    batch_data = batch_data[expected_cols]
                    
                    # Simulate batch processing with progress
                    batch_size = len(batch_data)
                    predictions = []
                    probabilities = []
                    
                    for i, (_, row) in enumerate(batch_data.iterrows()):
                        # Simulate processing time for demo
                        time.sleep(0.05)
                        
                        # Get prediction and probability
                        pred = model.predict(pd.DataFrame([row]))[0]
                        proba = model.predict_proba(pd.DataFrame([row]))[0]
                        
                        predictions.append(pred)
                        probabilities.append(proba[1] if pred == '>50K' else proba[0])
                        
                        # Update progress
                        progress = int((i + 1) / batch_size * 100)
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing {i+1}/{batch_size} records...")
                    
                    # Add results to dataframe
                    batch_data['Prediction'] = predictions
                    batch_data['Confidence'] = [f"{p*100:.1f}%" for p in probabilities]
                    
                    # Show results
                    st.success("🎉 Batch predictions complete!")
                    
                    # Summary stats
                    high_earners = sum(1 for p in predictions if p == '>50K')
                    st.metric("High Earners (>50K)", f"{high_earners} employees", 
                              f"{high_earners/len(predictions)*100:.1f}% of total")
                    
                    # Show sample of results
                    with st.expander("View Prediction Results", expanded=True):
                        st.dataframe(batch_data, use_container_width=True)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Prediction Distribution")
                        fig = px.pie(
                            names=['>50K', '≤50K'],
                            values=[high_earners, len(predictions)-high_earners],
                            color=['>50K', '≤50K'],
                            color_discrete_map={'>50K':'#4CCD99', '≤50K':'#FFC700'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### Age vs. Salary")
                        fig = px.box(
                            batch_data,
                            x='Prediction',
                            y='age',
                            color='Prediction',
                            color_discrete_map={'>50K':'#4CCD99', '≤50K':'#FFC700'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Download button
                    csv = batch_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "📥 Download Full Results", 
                        csv, 
                        file_name='salary_predictions.csv', 
                        mime='text/csv',
                        use_container_width=True
                    )

with tab3:
    st.markdown("## 📊 Salary Insights")
    
    # Sample data visualization (in a real app, you'd use your actual data)
    st.markdown("### Salary Distribution by Occupation")
    
    # Generate sample data for visualization
    occupations = [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
        "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
        "Protective-serv", "Armed-Forces"
    ]
    
    # Create synthetic data for visualization
    np.random.seed(42)
    data = []
    for occupation in occupations:
        n = np.random.randint(50, 200)
        ages = np.random.normal(loc=40, scale=10, size=n).clip(18, 65)
        hours = np.random.normal(loc=40, scale=5, size=n).clip(20, 60)
        
        # Higher probability of >50K for certain occupations
        if occupation in ["Exec-managerial", "Prof-specialty"]:
            prob = 0.6
        elif occupation in ["Tech-support", "Sales"]:
            prob = 0.3
        else:
            prob = 0.1
            
        salary = np.random.choice(['≤50K', '>50K'], size=n, p=[1-prob, prob])
        
        for age, hour, sal in zip(ages, hours, salary):
            data.append({
                'Occupation': occupation,
                'Age': int(age),
                'HoursPerWeek': int(hour),
                'Salary': sal
            })
    
    viz_df = pd.DataFrame(data)
    
    # Visualization
    fig = px.histogram(
        viz_df,
        x='Occupation',
        color='Salary',
        barmode='group',
        color_discrete_map={'≤50K': '#FFC700', '>50K': '#4CCD99'},
        title="Salary Distribution Across Different Occupations"
    )
    fig.update_layout(xaxis_title="Occupation", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
    
    # Age vs Salary
    st.markdown("### Age Distribution by Salary Class")
    fig = px.violin(
        viz_df,
        x='Salary',
        y='Age',
        color='Salary',
        color_discrete_map={'≤50K': '#FFC700', '>50K': '#4CCD99'},
        box=True,
        points="all"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Hours vs Salary
    st.markdown("### Weekly Hours vs Salary Class")
    fig = px.scatter(
        viz_df,
        x='HoursPerWeek',
        y='Age',
        color='Salary',
        color_discrete_map={'≤50K': '#FFC700', '>50K': '#4CCD99'},
        marginal_x="box",
        marginal_y="box"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key takeaways
    st.markdown("### 🔑 Key Insights")
    st.success("""
    - **Executive/Managerial** and **Professional Specialty** roles have the highest proportion of >50K earners
    - Employees earning >50K tend to be older (typically 35+ years)
    - Higher weekly working hours correlate with higher salary potential
    - Education level (not shown here) is a strong predictor of salary class
    """)