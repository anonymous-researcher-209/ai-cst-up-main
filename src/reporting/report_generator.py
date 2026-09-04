"""Report generation orchestration"""

import os
from datetime import datetime
from .formatters import TextFormatter, CSVFormatter, JSONFormatter, MarkdownFormatter
from .html_reporter import HTMLReportGenerator


class ReportGenerator:
    """Generate reports in multiple formats"""
    
    def __init__(self, output_dir="results"):
        """
        Initialize ReportGenerator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(output_dir, self.timestamp)
    
    def generate_all(self, results, single_model=False):
        """
        Generate all report formats
        
        Args:
            results: Evaluation results (single result dict or list)
            single_model: True if evaluating single model
            
        Returns:
            dict: Paths to generated report files
        """
        # Ensure output directory exists
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Normalize results to list
        if single_model and not isinstance(results, list):
            results = [results]
        
        report_paths = {}
        
        # Generate text report
        text_path = os.path.join(self.run_dir, "report.txt")
        self._generate_text_report(results, text_path, single_model)
        report_paths['text'] = text_path
        
        # Generate CSV reports
        csv_summary_path = os.path.join(self.run_dir, "summary.csv")
        csv_detailed_path = os.path.join(self.run_dir, "detailed.csv")
        CSVFormatter.format_summary(results, csv_summary_path)
        CSVFormatter.format_detailed(results, csv_detailed_path)
        report_paths['csv_summary'] = csv_summary_path
        report_paths['csv_detailed'] = csv_detailed_path
        
        # Generate JSON report
        json_path = os.path.join(self.run_dir, "results.json")
        JSONFormatter.format_results(results, json_path)
        report_paths['json'] = json_path
        
        # Generate Markdown report (for comparison)
        if not single_model:
            md_path = os.path.join(self.run_dir, "comparison.md")
            MarkdownFormatter.format_comparison(results, md_path)
            report_paths['markdown'] = md_path
        
        # Generate HTML report (interactive with charts)
        html_reporter = HTMLReportGenerator(self.run_dir)
        html_results = self._prepare_html_results(results)
        html_path = html_reporter.generate(html_results)
        if html_path:
            report_paths['html'] = html_path
        
        # Generate metadata
        metadata_path = os.path.join(self.run_dir, "metadata.json")
        self._generate_metadata(results, metadata_path, single_model)
        report_paths['metadata'] = metadata_path
        
        return report_paths
    
    def _generate_text_report(self, results, output_path, single_model):
        """Generate text report"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("HINDI MEDICAL AUDIO ASR EVALUATION REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 120 + "\n\n")
            
            if single_model:
                # Single model report
                if results:
                    f.write(TextFormatter.format_single_result(results[0]))
            else:
                # Comparison report
                f.write(TextFormatter.format_comparison(results))
                f.write("\n\n")
                
                # Detailed results for each model
                f.write("=" * 120 + "\n")
                f.write("DETAILED PREDICTIONS BY MODEL\n")
                f.write("=" * 120 + "\n\n")
                
                for result in results:
                    if result and result['files_processed'] > 0:
                        f.write(TextFormatter.format_single_result(result))
                        f.write("\n\n")
    
    def _generate_metadata(self, results, output_path, single_model):
        """Generate metadata file"""
        import json
        from ..utils.device_utils import get_device_info
        
        metadata = {
            'timestamp': self.timestamp,
            'generated_at': datetime.now().isoformat(),
            'evaluation_type': 'single_model' if single_model else 'comparison',
            'num_models': len(results) if results else 0,
            'device_info': get_device_info(),
            'models_evaluated': []
        }
        
        for result in results:
            if result:
                metadata['models_evaluated'].append({
                    'name': result['model_name'],
                    'model_id': result['model_id'],
                    'files_processed': result['files_processed']
                })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def print_summary(self, report_paths):
        """Print summary of generated reports"""
        print(f"\n📊 Reports generated in: {self.run_dir}")
        print(f"   📄 Text report: {os.path.basename(report_paths['text'])}")
        print(f"   📊 CSV summary: {os.path.basename(report_paths['csv_summary'])}")
        print(f"   📊 CSV detailed: {os.path.basename(report_paths['csv_detailed'])}")
        print(f"   📋 JSON results: {os.path.basename(report_paths['json'])}")
        if 'markdown' in report_paths:
            print(f"   📝 Markdown: {os.path.basename(report_paths['markdown'])}")
        if 'html' in report_paths:
            print(f"   🌐 HTML report: {os.path.basename(report_paths['html'])}")
        print(f"   ℹ️  Metadata: {os.path.basename(report_paths['metadata'])}")
    
    def _prepare_html_results(self, results):
        """Prepare results in format expected by HTML reporter"""
        html_results = []
        for result in results:
            if result is None:
                continue
            
            html_result = {
                'model_name': result.get('model_name', 'Unknown'),
                'model_id': result.get('model_id', 'N/A'),
                'model_type': result.get('model_type', 'unknown'),
                'avg_wer': result.get('avg_wer', 0) or 0,
                'avg_cer': result.get('avg_cer', 0) or 0,
                'avg_inference_time': result.get('avg_inference_time', 0) or 0,
                'total_time': result.get('total_time', 0),
                'files_processed': result.get('files_processed', 0),
                'parameters_M': result.get('parameters_M', 0),
                'per_file_results': []
            }
            
            # Add per-file results for detailed view
            predictions = result.get('predictions', {})
            for audio_path, pred_data in predictions.items():
                filename = os.path.basename(audio_path)
                html_result['per_file_results'].append({
                    'audio_file': filename,
                    'ground_truth': pred_data.get('actual', ''),
                    'prediction': pred_data.get('predicted', ''),
                    'wer': pred_data.get('wer', 0),
                    'cer': pred_data.get('cer', 0),
                    'inference_time': pred_data.get('inference_time', 0)
                })
            
            html_results.append(html_result)
        
        return html_results
