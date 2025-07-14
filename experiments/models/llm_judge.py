import pandas as pd
import os
import json
from .providers.openai import OpenAIProvider
import time
import concurrent.futures

class LLMJudge:
    def __init__(self, prompt: str, model_name: str = "gpt-4.1-2025-04-14", output_column: str = "output", question_column: str = "question", correct_answer_column: str = "answer", distractors_file: str = None):
        self.prompt = prompt
        self.model_name = model_name
        self.output_column = output_column
        self.question_column = question_column
        self.correct_answer_column = correct_answer_column
        self.provider = OpenAIProvider()
        self.distractors_text = self._load_distractors(distractors_file) if distractors_file else ""
    
    def _load_distractors(self, distractors_file: str) -> str:
        with open(distractors_file, 'r') as f:
            distractors_data = json.load(f)
        
        distractors_list = []
        for key in sorted(distractors_data.keys()):
            distractor_text = distractors_data[key]["rewrite_for_analysis"]
            distractors_list.append(distractor_text)
        
        formatted_distractors = []
        for i, distractor in enumerate(distractors_list):
            formatted_distractors.append(f"{i}. {distractor}")
        
        return "\n".join(formatted_distractors)
    
    def _format_prompt(self, output_value: str, question: str, correct_answer: str) -> str:
        if self.distractors_text:
            return self.prompt.format(output=output_value, question=question, correct_answer=correct_answer, distractors=self.distractors_text)
        else:
            return self.prompt.format(output=output_value, question=question, correct_answer=correct_answer)
    
    def _process_for_evaluation(self, input_df: pd.DataFrame, output_df: pd.DataFrame, indices_to_process: list[int], output_path: str, output_column_name: str = "llm_judge_output") -> None:
        timeout_per_request = 500
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices_to_process)) as executor:
            futures = {
                executor.submit(
                    self.provider.process_single_prompt,
                    prompt=self._format_prompt(
                        str(input_df.loc[idx, self.output_column]),
                        str(input_df.loc[idx, self.question_column]),
                        str(input_df.loc[idx, self.correct_answer_column])
                    ),
                    model_name=self.model_name,
                    max_output_tokens=100,
                    index=int(idx),
                ): idx 
                for idx in indices_to_process
            }
            
            for future in futures:
                idx = futures[future]
                try:
                    idx_result, response = future.result(timeout=timeout_per_request)
                    output_df.loc[idx_result, output_column_name] = response
                    
                    success = not response.startswith('ERROR')
                    status = "Success" if success else "Error"
                    print(f"{status} Row {idx_result}: {response}...")
                    
                except concurrent.futures.TimeoutError:
                    print(f"Row {idx}: Request timed out after {timeout_per_request}s - marking as timeout error")
                    output_df.loc[idx, output_column_name] = f"ERROR_TIMEOUT: Request exceeded {timeout_per_request}s"
                    
                except Exception as e:
                    output_df.loc[idx, output_column_name] = f"FUTURE_ERROR: {str(e)}"
                    print(f"Error Row {idx}: Future error: {e}")
        
        output_df.to_csv(output_path, index=False)
        
        completed_in_batch = len([idx for idx in indices_to_process if not pd.isna(output_df.loc[idx, output_column_name])])
        total_completed = (~output_df[output_column_name].isna() & 
                          ~output_df[output_column_name].str.startswith('ERROR', na=False)).sum()
        
        print(f"Saved progress: {completed_in_batch}/{len(indices_to_process)} in this batch")
        print(f"Overall progress: {total_completed}/{len(output_df)} ({total_completed/len(output_df)*100:.1f}%)")
        
        print(f"Results saved to: {output_path}")
        print(f"Successful: {total_completed}")
        print(f"Errors/Missing: {len(output_df) - total_completed}")
    
    def evaluate(self, input_path: str, output_path: str, max_context_length: int, max_tokens_per_minute: int, output_column_name: str = "llm_judge_output") -> None:
        input_df = pd.read_csv(input_path)
        input_df['token_count'] = [100] * len(input_df)
        
        if os.path.exists(output_path):
            print(f"Loading existing progress from {output_path}")
            output_df = pd.read_csv(output_path)
            
            if output_column_name not in output_df.columns:
                output_df[output_column_name] = None
        else:
            output_df = input_df.copy()
            output_df['llm_judge_output'] = None

        need_processing = (
            output_df['llm_judge_output'].isna() | 
            output_df['llm_judge_output'].astype(str).str.contains('ERROR', na=False)
        )
        to_process = output_df[need_processing].index.tolist()
        
        if to_process:
            print(f"{len(to_process)} rows needing processing: {to_process[0]} to {to_process[-1]}")
        else:
            print("All rows already processed successfully")
            return
            
        input_to_process = input_df.loc[to_process]
        batches = self.provider.create_batches(input_to_process, max_tokens_per_minute)
        print(f"Created {len(batches)} batches based on {max_tokens_per_minute:,} tokens/minute")
        
        for i, batch_indices in enumerate(batches):
            self._process_for_evaluation(input_to_process, output_df, batch_indices, output_path)
            if i < len(batches) - 1:
                print("Waiting 60 seconds")
                time.sleep(60)

    def analyze_distractors(self, input_path: str, output_path: str, max_context_length: int, max_tokens_per_minute: int, output_column_name: str = "distractor_label") -> pd.DataFrame:
        input_df = pd.read_csv(input_path)

        input_df_filtered = input_df[input_df['token_count'] <= max_context_length].copy()
        input_df_filtered = input_df_filtered[input_df_filtered['llm_judge_output'] == False]

        if os.path.exists(output_path):
            print(f"Loading existing progress from {output_path}")
            output_df = pd.read_csv(output_path)
            
            if output_column_name not in output_df.columns:
                output_df[output_column_name] = None
        else:
            output_df = input_df_filtered.copy()
            output_df[output_column_name] = None

        need_processing = (
            output_df[output_column_name].isna() | 
            output_df[output_column_name].astype(str).str.contains('ERROR', na=False)
        )
        to_process = output_df[need_processing].index.tolist()
        
        if to_process:
            print(f"{len(to_process)} rows needing processing: {to_process[0]} to {to_process[-1]}")
        else:
            print("All rows already processed successfully")
            return output_df
            
        input_to_process = input_df_filtered.loc[to_process]
        batches = self.provider.create_batches(input_to_process, max_tokens_per_minute)
        print(f"Created {len(batches)} batches based on {max_tokens_per_minute:,} tokens/minute")
        
        for i, batch_indices in enumerate(batches):
            self._process_for_evaluation(input_to_process, output_df, batch_indices, output_path, output_column_name)
            if i < len(batches) - 1:
                print("Waiting 60 seconds")
                time.sleep(60)