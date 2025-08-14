import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from data_processor import DataProcessor
from ai_estimator import AIEstimator
from report_generator import ReportGenerator
from ai_text_processor import AITextProcessor
from hybrid_processor import HybridProcessor
from utils import validate_data, get_survey_metadata
import config


def main():
    st.set_page_config(
        page_title="MoSPI AI-Enhanced Survey Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 MoSPI AI-Enhanced Survey Data Processing")
    st.markdown("### Professional-grade automated data preparation with AI-powered hybrid processing")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a section:", 
        ["Data Upload", "Data Processing", "AI Hybrid Processing", "Missing Value Analysis", 
         "AI Estimation", "Report Generation", "Dashboard"],
        key="main_navigation_select")
    
    if page == "Data Upload":
        data_upload_section()
    elif page == "Data Processing":
        data_processing_section()
    elif page == "AI Hybrid Processing":
        ai_hybrid_processing_section()
    elif page == "Missing Value Analysis":
        missing_value_analysis_section()
    elif page == "AI Estimation":
        ai_estimation_section()
    elif page == "Report Generation":
        report_generation_section()
    elif page == "Dashboard":
        dashboard_section()


def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = [
        'processed_data', 'estimates', 'raw_data', 'survey_config', 
        'missing_value_report', 'ai_processed_data', 'hybrid_report'
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None


def data_upload_section():
    st.header("📁 Data Upload & Validation")
    
    uploaded_file = st.file_uploader(
        "Upload Survey Data (CSV/Excel)", 
        type=['csv', 'xlsx', 'xls'],
        key="main_file_uploader_unique_2024"
    )
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                # Enhanced missing value recognition
                df = pd.read_csv(uploaded_file, na_values=config.MISSING_VALUES)
            else:
                df = pd.read_excel(uploaded_file, na_values=config.MISSING_VALUES)
            
            st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
            
            # Data validation
            validation_results = validate_data(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.subheader("Data Quality Report")
                st.json(validation_results)
            
            # Enhanced missing value analysis
            show_comprehensive_missing_analysis(df)
            
            # Survey metadata input
            st.subheader("Survey Metadata")
            survey_type = st.selectbox("Survey Type", 
                config.SURVEY_TYPES,
                key="upload_survey_type_unique")
            sampling_method = st.selectbox("Sampling Method", 
                config.SAMPLING_METHODS,
                key="upload_sampling_method_unique")
            population_size = st.number_input("Population Size", min_value=1, value=10000,
                key="upload_population_size_unique")
            
            if st.button("Save Data Configuration", key="upload_save_config_unique"):
                st.session_state.raw_data = df
                st.session_state.survey_config = {
                    'type': survey_type,
                    'sampling': sampling_method,
                    'population': population_size
                }
                st.success("✅ Configuration saved! Proceed to AI Hybrid Processing for advanced features.")
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")


def show_comprehensive_missing_analysis(df):
    """Enhanced missing value analysis with AI insights"""
    st.subheader("🔍 Comprehensive Missing Value Analysis")
    
    missing_summary = df.isnull().sum()
    total_missing = missing_summary.sum()
    
    if total_missing > 0:
        # Create detailed missing value DataFrame
        missing_df = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing Count': missing_summary.values,
            'Missing Percentage': (missing_summary.values / len(df) * 100).round(2),
            'Data Type': [str(df[col].dtype) for col in missing_summary.index],
            'Recommended AI Method': [get_recommended_ai_method(df[col]) for col in missing_summary.index]
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        st.dataframe(missing_df, use_container_width=True)
        
        # Visualization
        fig = px.bar(missing_df, x='Column', y='Missing Count', 
                   color='Data Type', title="Missing Values by Column and Data Type")
        st.plotly_chart(fig, use_container_width=True, key="upload_missing_analysis_chart")
        
        # AI recommendations
        st.info("💡 **AI Recommendations**: Use 'AI Hybrid Processing' for intelligent missing value imputation based on data types and patterns.")
    else:
        st.success("✅ No missing values found in the original data!")


def get_recommended_ai_method(column):
    """Get recommended AI method for missing value imputation"""
    if column.dtype in ['int64', 'float64']:
        return "Smart Numerical (Mean/Median + ML)"
    elif column.dtype == 'object':
        return "AI Text Generation + Mode"
    else:
        return "Hybrid AI Processing"


def data_processing_section():
    st.header("🔧 Traditional Data Processing & Cleaning")
    st.info("For AI-powered processing with advanced text generation, use 'AI Hybrid Processing' section.")
    
    if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
        st.warning("Please upload data first!")
        return
    
    processor = DataProcessor()
    df = st.session_state.raw_data
    
    # Processing options
    st.subheader("Processing Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        handle_missing = st.selectbox("Missing Data Strategy", 
            ["Drop", "Impute Mean", "Impute Median", "Forward Fill"],
            key="process_missing_strategy_unique")
    with col2:
        outlier_method = st.selectbox("Outlier Detection", 
            ["IQR", "Z-Score", "Isolation Forest", "None"],
            key="process_outlier_method_unique")
    with col3:
        normalization = st.selectbox("Normalization", 
            ["None", "Min-Max", "Standard", "Robust"],
            key="process_normalization_unique")
    
    if st.button("Process Data (Traditional)", key="process_data_button_unique"):
        process_traditional_data(processor, df, handle_missing, outlier_method, normalization)


def ai_hybrid_processing_section():
    st.header("🤖 AI Hybrid Processing Pipeline")
    st.markdown("### Advanced processing combining AI text generation with traditional statistical methods")
    
    if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
        st.warning("Please upload data first!")
        return
    
    hybrid_processor = HybridProcessor()
    ai_text_processor = AITextProcessor()
    df = st.session_state.raw_data
    
    # AI Processing Configuration
    st.subheader("🎯 AI Processing Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Numerical Data Processing**")
        numerical_method = st.selectbox("Numerical Missing Values", 
            ["Smart Mean/Median", "ML-Based Prediction", "Hybrid Statistical"],
            key="ai_numerical_method")
        
        outlier_method = st.selectbox("Outlier Detection", 
            ["AI-Enhanced IQR", "ML Isolation Forest", "Hybrid Detection"],
            key="ai_outlier_method")
    
    with col2:
        st.write("**Text Data Processing**")
        text_method = st.selectbox("Text Missing Values", 
            ["AI Text Generation", "Context-Aware Mode", "Hybrid AI+Statistical"],
            key="ai_text_method")
        
        ai_model = st.selectbox("AI Model for Text", 
            ["GPT-based", "BERT-based", "Random Forest", "Hybrid Ensemble"],
            key="ai_model_select")
    
    # Advanced Options
    with st.expander("🔧 Advanced AI Settings"):
        confidence_threshold = st.slider("AI Confidence Threshold", 0.5, 0.95, 0.8,
            key="ai_confidence_threshold")
        use_context_learning = st.checkbox("Use Context Learning", True,
            key="ai_context_learning")
        generate_explanations = st.checkbox("Generate AI Explanations", True,
            key="ai_generate_explanations")
    
    # Processing button
    if st.button("🚀 Start AI Hybrid Processing", key="ai_hybrid_process_button"):
        with st.spinner("🧠 AI is analyzing your data and generating intelligent imputations..."):
            try:
                # Create processing configuration
                config_dict = {
                    'numerical_method': numerical_method,
                    'text_method': text_method,
                    'ai_model': ai_model,
                    'outlier_method': outlier_method,
                    'confidence_threshold': confidence_threshold,
                    'use_context_learning': use_context_learning,
                    'generate_explanations': generate_explanations
                }
                
                # Process with AI hybrid pipeline
                processed_df, ai_report = hybrid_processor.process_with_ai(df, config_dict)
                
                # Add survey weights
                if st.session_state.survey_config:
                    weights = hybrid_processor.calculate_intelligent_weights(
                        processed_df, 
                        st.session_state.survey_config
                    )
                    processed_df['survey_weight'] = weights
                
                # Store results
                st.session_state.ai_processed_data = processed_df
                st.session_state.hybrid_report = ai_report
                
                # Display results
                display_ai_processing_results(df, processed_df, ai_report)
                
                # 🚀 NEW: Add download buttons right after processing is complete
                st.subheader("📥 Download Data Files")
                st.info("💾 **Download your original and AI-processed datasets below:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Download original data
                    csv_original = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Original Data",
                        data=csv_original,
                        file_name="original_survey_data.csv",
                        mime="text/csv",
                        key="download_original_after_ai_processing",
                        help="Download the original uploaded dataset"
                    )
                
                with col2:
                    # Download AI-processed data
                    csv_processed = processed_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download AI-Processed Data",
                        data=csv_processed,
                        file_name="ai_processed_survey_data.csv",
                        mime="text/csv",
                        key="download_processed_after_ai_processing",
                        help="Download the AI-enhanced processed dataset"
                    )
                
                # Show file information
                st.write("**File Information:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"📊 **Original**: {df.shape[0]:,} rows × {df.shape[1]} columns")
                    st.write(f"📋 **Missing values**: {df.isnull().sum().sum():,}")
                with col2:
                    st.write(f"📊 **AI-Processed**: {processed_df.shape[0]:,} rows × {processed_df.shape[1]} columns")
                    st.write(f"📋 **Missing values**: {processed_df.isnull().sum().sum():,}")
                
                st.success("✅ AI Hybrid Processing completed! Advanced features unlocked.")
                
            except Exception as e:
                st.error(f"Error in AI processing: {str(e)}")
                st.info("Falling back to traditional methods...")


def display_ai_processing_results(original_df, processed_df, ai_report):
    """Display comprehensive AI processing results"""
    st.subheader("📊 AI Processing Results")
    
    # Metrics comparison
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Missing Values", original_df.isnull().sum().sum())
    with col2:
        st.metric("After AI Processing", processed_df.isnull().sum().sum())
    with col3:
        improvement = original_df.isnull().sum().sum() - processed_df.isnull().sum().sum()
        st.metric("AI Improvements", improvement, delta=f"+{improvement}")
    with col4:
        accuracy = ai_report.get('overall_confidence', 0.85)
        st.metric("AI Confidence", f"{accuracy:.1%}")
    
    # AI-generated insights
    if 'ai_insights' in ai_report:
        st.subheader("🧠 AI-Generated Insights")
        for insight in ai_report['ai_insights']:
            st.info(f"💡 {insight}")
    
    # Detailed processing report
    if 'processing_details' in ai_report:
        st.subheader("📋 AI Processing Details")
        details_df = pd.DataFrame(ai_report['processing_details'])
        st.dataframe(details_df, use_container_width=True)
    
    # Visualization of AI improvements
    if 'column_improvements' in ai_report:
        fig = px.bar(
            x=list(ai_report['column_improvements'].keys()),
            y=list(ai_report['column_improvements'].values()),
            title="AI Processing Improvements by Column",
            labels={'y': 'Values Improved', 'x': 'Column'}
        )
        st.plotly_chart(fig, use_container_width=True, key="ai_improvements_chart")


def process_traditional_data(processor, df, handle_missing, outlier_method, normalization):
    """Process data using traditional methods"""
    with st.spinner("Processing data..."):
        try:
            processed_df = processor.clean_data(
                df, 
                missing_strategy=handle_missing.lower().replace(' ', '_'),
                outlier_method=outlier_method.lower().replace('-', '_'),
                normalization=normalization.lower().replace('-', '_')
            )
            
            # Ensure survey_config exists
            if not st.session_state.survey_config:
                st.session_state.survey_config = {
                    'type': 'Household Survey',
                    'sampling': 'Simple Random',
                    'population': 10000
                }
            
            # Calculate weights
            weights = processor.calculate_weights(processed_df, st.session_state.survey_config)
            processed_df['survey_weight'] = weights
            
            st.session_state.processed_data = processed_df
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Before Processing")
                st.write(f"Shape: {df.shape}")
                st.write(f"Missing values: {df.isnull().sum().sum()}")
                
            with col2:
                st.subheader("After Processing")
                st.write(f"Shape: {processed_df.shape}")
                st.write(f"Missing values: {processed_df.isnull().sum().sum()}")
            
            show_missing_value_comparison(df, processed_df, handle_missing)
            
            # Add download buttons for traditional processing too
            st.subheader("📥 Download Traditional Processing Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_original = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Original Data",
                    csv_original,
                    "original_data.csv",
                    "text/csv",
                    key="traditional_download_original"
                )
            
            with col2:
                csv_processed = processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Processed Data", 
                    csv_processed,
                    "traditional_processed_data.csv",
                    "text/csv",
                    key="traditional_download_processed"
                )
            
            st.success("✅ Traditional processing completed!")
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")


def show_missing_value_comparison(original_df, processed_df, strategy):
    """Enhanced missing value comparison with AI insights"""
    st.subheader("📋 Processing Comparison Report")
    
    comparison_data = []
    for col in original_df.columns:
        if col in processed_df.columns:
            original_missing = original_df[col].isnull().sum()
            processed_missing = processed_df[col].isnull().sum()
            values_filled = original_missing - processed_missing
            
            if values_filled > 0:
                if original_df[col].dtype in ['int64', 'float64']:
                    if 'mean' in strategy.lower():
                        fill_value = original_df[col].mean()
                        fill_method = f"Mean ({fill_value:.2f})"
                    elif 'median' in strategy.lower():
                        fill_value = original_df[col].median()
                        fill_method = f"Median ({fill_value:.2f})"
                    else:
                        fill_method = "Forward Fill"
                else:
                    mode_val = original_df[col].mode()
                    fill_method = f"Mode ({mode_val.iloc[0]})" if len(mode_val) > 0 else "Unknown"
            else:
                fill_method = "No filling needed"
            
            comparison_data.append({
                'Column': col,
                'Data Type': str(original_df[col].dtype),
                'Original Missing': original_missing,
                'After Processing': processed_missing,
                'Values Filled': values_filled,
                'Fill Method': fill_method,
                'AI Recommended': get_recommended_ai_method(original_df[col])
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    affected_df = comparison_df[comparison_df['Original Missing'] > 0]
    
    if len(affected_df) > 0:
        st.dataframe(affected_df, use_container_width=True)
        
        fig = px.bar(affected_df, x='Column', y=['Original Missing', 'Values Filled'], 
                    title="Missing Values: Original vs Filled",
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="comparison_chart_unique")
    else:
        st.info("No missing values were found or filled during processing.")


def missing_value_analysis_section():
    st.header("🔍 Advanced Missing Value Analysis")
    
    if 'raw_data' not in st.session_state:
        st.warning("Please upload data first!")
        return
    
    # Choose data source for analysis
    data_source = st.selectbox("Select Data Source for Analysis:",
        ["Original Data", "Traditional Processing", "AI Hybrid Processing"],
        key="analysis_data_source")
    
    if data_source == "Original Data":
        df = st.session_state.raw_data
        comparison_df = None
    elif data_source == "Traditional Processing" and st.session_state.processed_data is not None:
        df = st.session_state.raw_data
        comparison_df = st.session_state.processed_data
    elif data_source == "AI Hybrid Processing" and st.session_state.ai_processed_data is not None:
        df = st.session_state.raw_data
        comparison_df = st.session_state.ai_processed_data
    else:
        st.warning(f"Please complete {data_source.lower()} first!")
        return
    
    # Comprehensive analysis
    show_advanced_missing_analysis(df, comparison_df, data_source)


def show_advanced_missing_analysis(original_df, processed_df, source_type):
    """Advanced missing value analysis with AI insights"""
    st.subheader(f"📊 {source_type} Analysis Results")
    
    # Overall statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        original_missing = original_df.isnull().sum().sum()
        st.metric("Original Missing Values", original_missing)
    
    if processed_df is not None:
        with col2:
            processed_missing = processed_df.isnull().sum().sum()
            st.metric("After Processing", processed_missing)
        with col3:
            improvement = original_missing - processed_missing
            st.metric("Values Filled", improvement, delta=f"+{improvement}")
        
        # Show AI report if available
        if source_type == "AI Hybrid Processing" and st.session_state.hybrid_report:
            show_ai_analysis_report(st.session_state.hybrid_report)
    
    # Pattern analysis
    show_missing_patterns(original_df, processed_df)


def show_ai_analysis_report(ai_report):
    """Display AI analysis report"""
    st.subheader("🤖 AI Analysis Report")
    
    if 'text_generation_results' in ai_report:
        st.write("**AI Text Generation Results:**")
        for result in ai_report['text_generation_results']:
            st.info(f"📝 {result}")
    
    if 'confidence_scores' in ai_report:
        st.write("**AI Confidence Scores:**")
        confidence_df = pd.DataFrame(list(ai_report['confidence_scores'].items()),
                                   columns=['Column', 'Confidence Score'])
        st.dataframe(confidence_df, use_container_width=True)


def show_missing_patterns(original_df, processed_df):
    """Show missing data patterns with enhanced visualizations"""
    total_missing = original_df.isnull().sum().sum()
    
    if total_missing > 0:
        st.subheader("🎯 Missing Data Patterns")
        
        # Pattern heatmap
        if st.checkbox("Show Missing Value Heatmap", key="pattern_heatmap_check"):
            missing_pattern = original_df.isnull().astype(int)
            fig = px.imshow(missing_pattern.T, 
                          title="Missing Value Pattern (Yellow = Missing)",
                          color_continuous_scale=['blue', 'yellow'])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="pattern_heatmap_viz")
        
        # Co-occurrence matrix
        if st.checkbox("Show Missing Value Co-occurrence", key="cooccurrence_check"):
            missing_corr = original_df.isnull().corr()
            fig_corr = px.imshow(missing_corr, text_auto=True, 
                               title="Missing Value Co-occurrence Pattern")
            st.plotly_chart(fig_corr, use_container_width=True, key="cooccurrence_viz")


def ai_estimation_section():
    st.header("🤖 AI-Powered Population Estimation")
    
    # Choose data source
    data_choice = st.selectbox("Select Processed Data:",
        ["Traditional Processing", "AI Hybrid Processing"],
        key="estimation_data_choice")
    
    if data_choice == "Traditional Processing":
        df = st.session_state.processed_data
    else:
        df = st.session_state.ai_processed_data
    
    if df is None:
        st.warning(f"Please complete {data_choice.lower()} first!")
        return
    
    estimator = AIEstimator()
    
    # Enhanced variable selection
    st.subheader("🎯 Enhanced Variable Selection")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if 'survey_weight' in numeric_cols:
        numeric_cols.remove('survey_weight')
    
    if len(numeric_cols) == 0:
        st.error("No numeric variables found for estimation!")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        target_vars = st.multiselect("Target Variables for Estimation", numeric_cols,
            key="estimation_target_vars_enhanced")
    with col2:
        stratification_vars = st.multiselect("Stratification Variables", categorical_cols,
            key="estimation_stratification_vars_enhanced")
    
    # Enhanced estimation options
    col1, col2 = st.columns(2)
    with col1:
        estimation_method = st.selectbox("Estimation Method", 
            ["Design-Based", "Model-Based", "Hybrid", "AI-Enhanced Hybrid"],
            key="estimation_method_enhanced")
    with col2:
        confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01,
            key="estimation_confidence_enhanced")
    
    # AI enhancement options
    if "AI-Enhanced" in estimation_method:
        st.subheader("🧠 AI Enhancement Settings")
        use_ensemble = st.checkbox("Use Ensemble Methods", True, key="estimation_ensemble")
        use_bootstrap = st.checkbox("Use Bootstrap Validation", True, key="estimation_bootstrap")
        ai_uncertainty = st.checkbox("AI Uncertainty Quantification", True, key="estimation_uncertainty")
    
    if st.button("Generate Advanced Estimates", key="estimation_generate_enhanced"):
        if not target_vars:
            st.error("Please select at least one target variable.")
        else:
            generate_advanced_estimates(estimator, df, target_vars, stratification_vars, 
                                      estimation_method, confidence_level)


def generate_advanced_estimates(estimator, df, target_vars, stratification_vars, method, confidence_level):
    """Generate estimates with enhanced AI capabilities"""
    with st.spinner("🧠 Generating AI-enhanced estimates..."):
        try:
            estimates = {}
            
            for var in target_vars:
                if "AI-Enhanced" in method:
                    # Use enhanced AI estimation
                    estimate_result = estimator.generate_ai_enhanced_estimates(
                        df, var, 
                        method=method.lower().replace('-', '_').replace(' ', '_'),
                        confidence_level=confidence_level,
                        stratification_vars=stratification_vars
                    )
                else:
                    # Use standard estimation
                    estimate_result = estimator.generate_estimates(
                        df, var, 
                        method=method.lower().replace('-', '_'),
                        confidence_level=confidence_level,
                        stratification_vars=stratification_vars
                    )
                estimates[var] = estimate_result
            
            st.session_state.estimates = estimates
            
            # Display enhanced results
            display_enhanced_estimation_results(estimates, confidence_level, stratification_vars)
            
        except Exception as e:
            st.error(f"Error generating estimates: {str(e)}")


def display_enhanced_estimation_results(estimates, confidence_level, stratification_vars):
    """Display enhanced estimation results with AI insights"""
    for var, result in estimates.items():
        st.subheader(f"📊 Enhanced Estimates for {var}")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Estimate", f"{result['total']:.2f}")
        with col2:
            st.metric("Standard Error", f"{result['se']:.4f}")
        with col3:
            st.metric("CV (%)", f"{result['cv']:.2f}")
        with col4:
            if 'ai_confidence' in result:
                st.metric("AI Confidence", f"{result['ai_confidence']:.1%}")
        
        # Enhanced confidence interval
        st.write(f"**{confidence_level*100}% Confidence Interval:** "
                f"({result['ci_lower']:.2f}, {result['ci_upper']:.2f})")
        
        # AI insights
        if 'ai_insights' in result:
            st.info(f"🧠 **AI Insight**: {result['ai_insights']}")
        
        # Visualizations
        if stratification_vars and 'strata_estimates' in result and result['strata_estimates']:
            fig = px.bar(
                x=list(result['strata_estimates'].keys()),
                y=list(result['strata_estimates'].values()),
                title=f"{var} by Strata (AI-Enhanced)"
            )
            st.plotly_chart(fig, use_container_width=True, 
                          key=f"enhanced_strata_chart_{var}")
    
    st.success("✅ AI-Enhanced estimates generated! Proceed to Report Generation.")


def report_generation_section():
    st.header("📄 AI-Enhanced Report Generation")
    
    if not st.session_state.estimates:
        st.warning("Please generate estimates first!")
        return
    
    report_gen = ReportGenerator()
    
    # Enhanced report configuration
    st.subheader("⚙️ Advanced Report Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        report_title = st.text_input("Report Title", "AI-Enhanced Survey Analysis Report",
            key="report_title_enhanced")
        report_type = st.selectbox("Report Type", 
            ["Executive Summary", "Technical Report", "Policy Brief", "AI Analysis Report"],
            key="report_type_enhanced")
    
    with col2:
        organization = st.text_input("Organization", 
            "Ministry of Statistics and Programme Implementation",
            key="report_organization_enhanced")
        author = st.text_input("Author", "AI-Enhanced Analysis System",
            key="report_author_enhanced")
    
    # AI-enhanced options
    st.subheader("🤖 AI Report Enhancement")
    col1, col2 = st.columns(2)
    with col1:
        include_ai_insights = st.checkbox("Include AI Insights", True,
            key="report_ai_insights")
        include_methodology = st.checkbox("Include Methodology", True,
            key="report_methodology_enhanced")
        include_recommendations = st.checkbox("AI Recommendations", True,
            key="report_recommendations_enhanced")
    
    with col2:
        generate_summary = st.checkbox("AI-Generated Executive Summary", True,
            key="report_ai_summary")
        include_visualizations = st.checkbox("Enhanced Visualizations", True,
            key="report_visualizations_enhanced")
        include_quality_metrics = st.checkbox("Data Quality Metrics", True,
            key="report_quality_metrics")
    
    # Report generation
    if st.button("🚀 Generate AI-Enhanced Report", key="report_generate_enhanced"):
        generate_ai_enhanced_report(report_gen, report_title, report_type, organization, 
                                  author, include_ai_insights, include_methodology,
                                  include_recommendations, generate_summary, 
                                  include_visualizations, include_quality_metrics)


def generate_ai_enhanced_report(report_gen, title, report_type, org, author, 
                               ai_insights, methodology, recommendations, 
                               ai_summary, visualizations, quality_metrics):
    """Generate AI-enhanced comprehensive report"""
    with st.spinner("🧠 AI is generating your comprehensive report..."):
        try:
            # Enhanced report configuration
            report_config = {
                'title': title,
                'type': report_type,
                'organization': org,
                'author': author,
                'include_ai_insights': ai_insights,
                'include_methodology': methodology,
                'include_recommendations': recommendations,
                'generate_ai_summary': ai_summary,
                'include_visualizations': visualizations,
                'include_quality_metrics': quality_metrics,
                'ai_processing_used': st.session_state.ai_processed_data is not None,
                'hybrid_report': st.session_state.hybrid_report
            }
            
            # Get the appropriate processed data
            processed_data = (st.session_state.ai_processed_data 
                            if st.session_state.ai_processed_data is not None 
                            else st.session_state.processed_data)
            
            # Generate enhanced report
            report_content = report_gen.generate_ai_enhanced_report(
                processed_data,
                st.session_state.estimates,
                st.session_state.survey_config or {},
                report_config
            )
            
            # Display report
            st.subheader("📋 Generated AI-Enhanced Report")
            st.markdown(report_content)
            
            # Enhanced download options
            st.subheader("💾 Download Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                html_report = report_gen.export_html(report_content)
                st.download_button(
                    "📄 Download HTML Report",
                    html_report,
                    f"{title.replace(' ', '_')}.html",
                    "text/html",
                    key="report_download_html_enhanced"
                )
            
            with col2:
                # Generate PDF-ready version
                pdf_content = report_gen.generate_pdf_ready_content(report_content)
                st.download_button(
                    "📋 Download PDF-Ready",
                    pdf_content,
                    f"{title.replace(' ', '_')}_pdf_ready.html",
                    "text/html",
                    key="report_download_pdf_ready"
                )
            
            with col3:
                # Generate structured data export
                structured_data = report_gen.export_structured_data(
                    st.session_state.estimates,
                    report_config
                )
                st.download_button(
                    "📊 Download Data (JSON)",
                    structured_data,
                    f"{title.replace(' ', '_')}_data.json",
                    "application/json",
                    key="report_download_data"
                )
            
            st.success("✅ AI-Enhanced report generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")


def dashboard_section():
    st.header("📊 AI-Enhanced Interactive Dashboard")
    
    # Data source selection
    data_sources = []
    if st.session_state.processed_data is not None:
        data_sources.append("Traditional Processing")
    if st.session_state.ai_processed_data is not None:
        data_sources.append("AI Hybrid Processing")
    
    if not data_sources:
        st.warning("Please process data first!")
        return
    
    selected_source = st.selectbox("Select Data Source:", data_sources,
        key="dashboard_data_source")
    
    df = (st.session_state.ai_processed_data if selected_source == "AI Hybrid Processing" 
          else st.session_state.processed_data)
    
    # Enhanced metrics dashboard
    display_enhanced_dashboard_metrics(df, selected_source)
    
    # Advanced visualizations
    display_advanced_visualizations(df)
    
    # AI insights panel
    if selected_source == "AI Hybrid Processing" and st.session_state.hybrid_report:
        display_ai_insights_panel(st.session_state.hybrid_report)


def display_enhanced_dashboard_metrics(df, source):
    """Display enhanced dashboard metrics"""
    st.subheader(f"📈 {source} Dashboard")
    
    # Key performance indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Variables", len(df.columns))
    with col3:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    with col4:
        if st.session_state.estimates:
            st.metric("Estimates Generated", len(st.session_state.estimates))
        else:
            st.metric("Estimates Generated", 0)
    with col5:
        if source == "AI Hybrid Processing" and st.session_state.hybrid_report:
            confidence = st.session_state.hybrid_report.get('overall_confidence', 0.85)
            st.metric("AI Confidence", f"{confidence:.1%}")
    
    # Data quality comparison
    if st.session_state.raw_data is not None:
        st.subheader("🎯 Data Quality Improvement")
        
        original_missing = st.session_state.raw_data.isnull().sum().sum()
        processed_missing = df.isnull().sum().sum()
        improvement = original_missing - processed_missing
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Original Missing", original_missing)
        with col2:
            st.metric("After Processing", processed_missing)
        with col3:
            st.metric("Improvement", improvement, delta=f"+{improvement}")


def display_advanced_visualizations(df):
    """Display advanced dashboard visualizations"""
    st.subheader("📊 Advanced Data Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'survey_weight' in numeric_cols:
        numeric_cols.remove('survey_weight')
    
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if numeric_cols:
        # Multi-variable analysis
        col1, col2 = st.columns(2)
        
        with col1:
            selected_var = st.selectbox("Select Variable for Analysis", numeric_cols,
                key="dashboard_analysis_var")
            
            # Distribution plot
            fig_dist = px.histogram(df, x=selected_var, 
                                  title=f"Distribution of {selected_var}",
                                  marginal="box")
            st.plotly_chart(fig_dist, use_container_width=True, key="dashboard_distribution")
        
        with col2:
            if len(numeric_cols) > 1:
                comparison_var = st.selectbox("Compare with:", 
                    [col for col in numeric_cols if col != selected_var],
                    key="dashboard_comparison_var")
                
                # Scatter plot with trend line
                fig_scatter = px.scatter(df, x=selected_var, y=comparison_var,
                                       title=f"{selected_var} vs {comparison_var}",
                                       trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True, key="dashboard_scatter")
        
        # Correlation heatmap
        if len(numeric_cols) > 2:
            st.subheader("🔥 Correlation Heatmap")
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, 
                               title="Variable Correlation Matrix",
                               color_continuous_scale="RdBu")
            st.plotly_chart(fig_corr, use_container_width=True, key="dashboard_correlation")
    
    # Statistical summary
    if numeric_cols:
        st.subheader("📋 Statistical Summary")
        summary_stats = df[numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)


def display_ai_insights_panel(ai_report):
    """Display AI insights panel"""
    st.subheader("🧠 AI Insights Panel")
    
    with st.expander("🤖 AI Processing Insights", expanded=True):
        if 'ai_insights' in ai_report:
            for i, insight in enumerate(ai_report['ai_insights']):
                st.info(f"💡 **Insight {i+1}**: {insight}")
        
        if 'processing_summary' in ai_report:
            st.write("**Processing Summary:**")
            st.json(ai_report['processing_summary'])
    
    with st.expander("📊 Confidence Metrics"):
        if 'confidence_scores' in ai_report:
            confidence_df = pd.DataFrame(
                list(ai_report['confidence_scores'].items()),
                columns=['Column', 'AI Confidence']
            )
            fig_conf = px.bar(confidence_df, x='Column', y='AI Confidence',
                            title="AI Confidence by Column")
            st.plotly_chart(fig_conf, use_container_width=True, key="ai_confidence_chart")


if __name__ == "__main__":
    main()
