"""Report formatters for different output formats"""

import json
import csv
from datetime import datetime
from tabulate import tabulate


class TextFormatter:
    """Format results as plain text"""
    
    @staticmethod
    def format_single_result(result):
        """Format a single model result"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"Model: {result['model_name']}")
        lines.append(f"Model ID: {result['model_id']}")
        lines.append("=" * 80)
        lines.append("")
        
        if result['files_processed'] == 0:
            lines.append("No files processed successfully")
            return "\n".join(lines)
        
        lines.append(f"Parameters: {result['parameters_M']:.2f}M")
        lines.append(f"Load Time: {result['load_time']:.2f}s")
        lines.append(f"Average WER: {result['avg_wer']:.4f}")
        lines.append(f"Average CER: {result['avg_cer']:.4f}")
        lines.append(f"Average Inference Time: {result['avg_inference_time']:.2f}s")
        lines.append(f"Total Time: {result['total_time']:.2f}s")
        lines.append(f"Files Processed: {result['files_processed']}")
        lines.append("")
        
        lines.append("Detailed Predictions:")
        lines.append("-" * 80)
        for audio_file, pred_data in result['predictions'].items():
            filename = audio_file.split('\\')[-1]
            lines.append(f"\nFile: {filename}")
            
            # Show script info if available
            if 'predicted_script' in pred_data:
                lines.append(f"  [Script: Predicted={pred_data['predicted_script']}, Actual={pred_data['actual_script']}]")
            
            lines.append(f"  Predicted: {pred_data['predicted']}")
            
            # Show transliterated version if different from original
            if 'predicted_roman' in pred_data and pred_data['predicted_roman'] != pred_data['predicted']:
                lines.append(f"  Predicted (Roman): {pred_data['predicted_roman']}")
            
            lines.append(f"  Actual:    {pred_data['actual']}")
            
            # Show normalized versions used for comparison if available
            if 'normalized_predicted' in pred_data:
                lines.append(f"  [Normalized for WER: '{pred_data['normalized_predicted'][:50]}...' vs '{pred_data['normalized_actual'][:50]}...']")
            
            lines.append(f"  WER: {pred_data['wer']:.4f} | CER: {pred_data['cer']:.4f} | Time: {pred_data['inference_time']:.2f}s")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_comparison(results):
        """Format comparison of multiple models"""
        lines = []
        lines.append("=" * 120)
        lines.append("MODEL COMPARISON")
        lines.append("=" * 120)
        lines.append("")
        
        # Summary table
        table_data = []
        headers = ["Model", "Parameters (M)", "Avg WER", "Avg CER", "Avg Time (s)", "Total Time (s)", "Files"]
        
        valid_results = [r for r in results if r and r['files_processed'] > 0]
        
        for result in valid_results:
            table_data.append([
                result['model_name'],
                f"{result['parameters_M']:.1f}",
                f"{result['avg_wer']:.4f}",
                f"{result['avg_cer']:.4f}",
                f"{result['avg_inference_time']:.2f}",
                f"{result['total_time']:.2f}",
                result['files_processed']
            ])
        
        lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # Best models
        if valid_results:
            lines.append("")
            lines.append("🏆 BEST MODELS:")
            best_wer = min(valid_results, key=lambda x: x['avg_wer'])
            best_cer = min(valid_results, key=lambda x: x['avg_cer'])
            best_time = min(valid_results, key=lambda x: x['avg_inference_time'])
            
            lines.append(f"  • Best WER: {best_wer['model_name']} ({best_wer['avg_wer']:.4f})")
            lines.append(f"  • Best CER: {best_cer['model_name']} ({best_cer['avg_cer']:.4f})")
            lines.append(f"  • Fastest: {best_time['model_name']} ({best_time['avg_inference_time']:.2f}s)")
        
        return "\n".join(lines)


class CSVFormatter:
    """Format results as CSV"""
    
    @staticmethod
    def format_summary(results, output_file):
        """Write summary CSV file"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Model_ID', 'Parameters_M', 'Load_Time_s', 
                'Avg_WER', 'Avg_CER', 'Avg_Inference_Time_s', 
                'Total_Time_s', 'Files_Processed'
            ])
            
            for result in results:
                if result and result['files_processed'] > 0:
                    writer.writerow([
                        result['model_name'],
                        result['model_id'],
                        f"{result['parameters_M']:.2f}",
                        f"{result['load_time']:.2f}",
                        f"{result['avg_wer']:.4f}",
                        f"{result['avg_cer']:.4f}",
                        f"{result['avg_inference_time']:.2f}",
                        f"{result['total_time']:.2f}",
                        result['files_processed']
                    ])
    
    @staticmethod
    def format_detailed(results, output_file):
        """Write detailed per-file CSV"""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Model', 'Audio_File', 'Predicted', 'Predicted_Roman',
                'Actual', 'Predicted_Script', 'Actual_Script',
                'WER', 'CER', 'Inference_Time_s'
            ])
            
            for result in results:
                if result and result['files_processed'] > 0:
                    for audio_file, pred_data in result['predictions'].items():
                        filename = audio_file.split('\\')[-1]
                        writer.writerow([
                            result['model_name'],
                            filename,
                            pred_data['predicted'],
                            pred_data.get('predicted_roman', pred_data['predicted']),
                            pred_data['actual'],
                            pred_data.get('predicted_script', 'unknown'),
                            pred_data.get('actual_script', 'unknown'),
                            f"{pred_data['wer']:.4f}",
                            f"{pred_data['cer']:.4f}",
                            f"{pred_data['inference_time']:.2f}"
                        ])


class JSONFormatter:
    """Format results as JSON"""
    
    @staticmethod
    def format_results(results, output_file):
        """Write results as JSON"""
        # Make results JSON serializable
        json_results = []
        for result in results:
            if result:
                json_results.append(result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)


class MarkdownFormatter:
    """Format results as Markdown"""
    
    @staticmethod
    def format_comparison(results, output_file):
        """Write comparison as Markdown"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# ASR Model Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            
            # Summary table
            f.write("| Model | Parameters (M) | Avg WER | Avg CER | Avg Time (s) | Total Time (s) | Files |\n")
            f.write("|-------|----------------|---------|---------|--------------|----------------|-------|\n")
            
            valid_results = [r for r in results if r and r['files_processed'] > 0]
            
            for result in valid_results:
                f.write(f"| {result['model_name']} | ")
                f.write(f"{result['parameters_M']:.1f} | ")
                f.write(f"{result['avg_wer']:.4f} | ")
                f.write(f"{result['avg_cer']:.4f} | ")
                f.write(f"{result['avg_inference_time']:.2f} | ")
                f.write(f"{result['total_time']:.2f} | ")
                f.write(f"{result['files_processed']} |\n")
            
            # Best models
            if valid_results:
                f.write("\n## 🏆 Best Models\n\n")
                best_wer = min(valid_results, key=lambda x: x['avg_wer'])
                best_cer = min(valid_results, key=lambda x: x['avg_cer'])
                best_time = min(valid_results, key=lambda x: x['avg_inference_time'])
                
                f.write(f"- **Best WER:** {best_wer['model_name']} ({best_wer['avg_wer']:.4f})\n")
                f.write(f"- **Best CER:** {best_cer['model_name']} ({best_cer['avg_cer']:.4f})\n")
                f.write(f"- **Fastest:** {best_time['model_name']} ({best_time['avg_inference_time']:.2f}s)\n")
            
            # Detailed results
            f.write("\n## Detailed Results\n\n")
            for result in valid_results:
                f.write(f"\n### {result['model_name']}\n\n")
                f.write(f"- **Model ID:** {result['model_id']}\n")
                f.write(f"- **Parameters:** {result['parameters_M']:.2f}M\n")
                f.write(f"- **Load Time:** {result['load_time']:.2f}s\n\n")
                
                f.write("| File | Predicted | Actual | WER | CER | Time (s) |\n")
                f.write("|------|-----------|--------|-----|-----|----------|\n")
                
                for audio_file, pred_data in result['predictions'].items():
                    filename = audio_file.split('\\')[-1]
                    f.write(f"| {filename} | ")
                    f.write(f"{pred_data['predicted'][:50]}... | ")
                    f.write(f"{pred_data['actual'][:50]}... | ")
                    f.write(f"{pred_data['wer']:.4f} | ")
                    f.write(f"{pred_data['cer']:.4f} | ")
                    f.write(f"{pred_data['inference_time']:.2f} |\n")
