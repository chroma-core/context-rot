import pandas as pd
import time
import os
from typing import Any
import concurrent.futures
from abc import ABC, abstractmethod

class BaseProvider(ABC):
    def __init__(self):
        self.client = self.get_client()

    @abstractmethod
    def process_single_prompt(self, prompt: str, model_name: str, max_output_tokens: int, index: int) -> tuple[int, str]:
        pass

    @abstractmethod
    def get_client(self) -> Any:
        pass

    def create_batches(self, df: pd.DataFrame, max_tokens_per_minute: int) -> list[list[int]]:
        indices = df.index.tolist()
        token_counts = df['token_count'].tolist()
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx, tokens in zip(indices, token_counts):
            if tokens > max_tokens_per_minute:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
                
                batches.append([idx])
                continue
            
            if current_tokens + tokens > max_tokens_per_minute and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def process_batch(self, input_df: pd.DataFrame, output_df: pd.DataFrame, indices_to_process: list[int], model_name: str, output_path: str, input_column: str, output_column: str) -> None:
        timeout_per_request = 500
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices_to_process)) as executor:
            futures = {
                executor.submit(
                    self.process_single_prompt, 
                    prompt=str(input_df.loc[idx, input_column]), 
                    model_name=model_name,
                    max_output_tokens=int(input_df.loc[idx, 'max_output_tokens']) if 'max_output_tokens' in input_df.columns else 1000,
                    index=int(idx),
                ): idx 
                for idx in indices_to_process
            }
            
            for future in futures:
                idx = futures[future]
                try:
                    idx_result, response = future.result(timeout=timeout_per_request)
                    output_df.loc[idx_result, output_column] = response
                    
                    success = not response.startswith('ERROR')
                    status = "Success" if success else "Error"
                    print(f"{status} - Row {idx_result}: {response}...")
                    
                except concurrent.futures.TimeoutError:
                    print(f"Row {idx}: Request timed out after {timeout_per_request}s - marking as timeout error")
                    output_df.loc[idx, output_column] = f"ERROR_TIMEOUT: Request exceeded {timeout_per_request}s"
                    
                except Exception as e:
                    output_df.loc[idx, output_column] = f"FUTURE_ERROR: {str(e)}"
                    print(f"Error - Row {idx}: Future error: {e}")
        
        output_df.to_csv(output_path, index=False)
        
        completed_in_batch = len([idx for idx in indices_to_process if not pd.isna(output_df.loc[idx, output_column])])
        total_completed = (~output_df[output_column].isna() & 
                          ~output_df[output_column].str.startswith('ERROR', na=False)).sum()
        
        print(f"Saved progress: {completed_in_batch}/{len(indices_to_process)} in this batch")
        print(f"Overall progress: {total_completed}/{len(output_df)} ({total_completed/len(output_df)*100:.1f}%)")
        output_df.to_csv(output_path, index=False)
        
        print(f"Results saved to: {output_path}")
        print(f"Successful: {total_completed}")
        print(f"Errors/Missing: {len(output_df) - total_completed}")

    def main(self, input_path: str, output_path: str, input_column: str, output_column: str, model_name: str, max_context_length: int, max_tokens_per_minute: int) -> None:
        input_df = pd.read_csv(input_path)

        input_df_filtered = input_df[input_df['token_count'] <= max_context_length].copy()
        
        print(f"Filtered by max_context_length ({max_context_length:,} tokens): {len(input_df)} to {len(input_df_filtered)} rows ({len(input_df) - len(input_df_filtered)} filtered out)")

        if os.path.exists(output_path):
            print(f"Loading existing progress from {output_path}")
            output_df = pd.read_csv(output_path)
            
            if output_column not in output_df.columns:
                output_df[output_column] = None
        else:
            output_df = input_df_filtered.drop(columns=[input_column]).copy()
            output_df[output_column] = None

        need_processing = (
            output_df[output_column].isna() | 
            output_df[output_column].str.contains('ERROR', na=False)
        )
        to_process = output_df[need_processing].index.tolist()
        
        if to_process:
            print(f"{len(to_process)} rows needing processing: {to_process[0]} to {to_process[-1]}")
        else:
            print("All rows already processed successfully")
            return
            
        input_to_process = input_df_filtered.loc[to_process]
        batches = self.create_batches(input_to_process, max_tokens_per_minute)
        print(f"Created {len(batches)} batches based on {max_tokens_per_minute:,} tokens/minute")
        
        for i, batch_indices in enumerate(batches):
            self.process_batch(input_to_process, output_df, batch_indices, model_name, output_path, input_column, output_column)
            if i < len(batches) - 1:
                print("Waiting 60 seconds")
                time.sleep(60)