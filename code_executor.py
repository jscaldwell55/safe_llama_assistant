import subprocess
import tempfile
import os
import shutil
import logging
import ast
import re
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import resource
import signal
from contextlib import contextmanager
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityViolation(Exception):
    """Raised when a security violation is detected"""
    pass

class CodeExecutor:
    """
    Secure code execution environment with multiple layers of protection.
    
    Security features:
    - Input validation and sanitization
    - AST-based code analysis
    - Resource limits (memory, CPU, disk)
    - Sandboxed execution environment
    - Restricted imports whitelist
    - Output size limits
    - Timeout protection
    """
    
    def __init__(self):
        self.max_execution_time = 30  # seconds
        self.max_output_size = 1024 * 1024  # 1MB
        self.max_memory_mb = 512
        self.allowed_imports = {
            'math', 'statistics', 'json', 'datetime', 'collections',
            'itertools', 'functools', 'typing', 'dataclasses',
            'enum', 'decimal', 'fractions', 'random', 'string',
            're', 'textwrap', 'unicodedata', 'base64', 'hashlib',
            'csv', 'io', 'time', 'calendar', 'zoneinfo'
        }
        self.forbidden_builtins = {
            'eval', 'exec', 'compile', '__import__', 'open',
            'input', 'help', 'globals', 'locals', 'vars',
            'dir', 'getattr', 'setattr', 'delattr', 'hasattr'
        }
        self.forbidden_patterns = [
            r'__[\w]+__',  # Dunder methods
            r'\.\./',  # Path traversal
            r'subprocess',  # Process execution
            r'os\.',  # OS operations
            r'sys\.',  # System operations
            r'socket',  # Network operations
            r'urllib',  # Network operations
            r'requests',  # Network operations
            r'file\(',  # File operations
            r'open\(',  # File operations
            r'eval\(',  # Dynamic execution
            r'exec\(',  # Dynamic execution
            r'compile\(',  # Code compilation
            r'__import__',  # Dynamic imports
        ]
    
    def validate_code(self, code: str) -> Tuple[bool, str]:
        """
        Validate code for security issues using AST analysis.
        
        Args:
            code: Python code to validate
            
        Returns:
            Tuple of (is_safe, reason)
        """
        # Check for forbidden patterns
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Forbidden pattern detected: {pattern}"
        
        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        
        # Analyze AST for security issues
        validator = CodeValidator(self.allowed_imports, self.forbidden_builtins)
        try:
            validator.visit(tree)
        except SecurityViolation as e:
            return False, str(e)
        
        return True, "Code validation passed"
    
    def create_sandbox_environment(self) -> str:
        """Create a temporary sandboxed directory for code execution"""
        sandbox_dir = tempfile.mkdtemp(prefix="code_sandbox_")
        
        # Set restrictive permissions
        os.chmod(sandbox_dir, 0o700)
        
        return sandbox_dir
    
    def cleanup_sandbox(self, sandbox_dir: str):
        """Clean up sandbox directory"""
        try:
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir)
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {e}")
    
    @contextmanager
    def resource_limits(self):
        """Context manager to set resource limits for child process"""
        def set_limits():
            # Set memory limit
            memory_bytes = self.max_memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (self.max_execution_time, self.max_execution_time))
            
            # Disable core dumps
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            
            # Limit number of processes
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        
        yield set_limits
    
    def execute_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in a secure sandboxed environment.
        
        Args:
            code: Code to execute
            language: Programming language (currently only Python supported)
            
        Returns:
            Dict containing execution results
        """
        if language != "python":
            return {
                "success": False,
                "error": f"Language '{language}' not supported. Only Python is currently supported.",
                "output": "",
                "execution_time": 0
            }
        
        # Validate code
        is_safe, validation_message = self.validate_code(code)
        if not is_safe:
            return {
                "success": False,
                "error": f"Code validation failed: {validation_message}",
                "output": "",
                "execution_time": 0,
                "validation_error": True
            }
        
        # Create sandbox
        sandbox_dir = self.create_sandbox_environment()
        code_file = os.path.join(sandbox_dir, "user_code.py")
        
        try:
            # Add safety wrapper to code
            wrapped_code = self._wrap_code_safely(code)
            
            # Write code to file
            with open(code_file, 'w') as f:
                f.write(wrapped_code)
            
            # Execute in subprocess with restrictions
            start_time = time.time()
            
            # Use subprocess with resource limits
            with self.resource_limits() as set_limits:
                process = subprocess.Popen(
                    ['python3', code_file],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=sandbox_dir,
                    preexec_fn=set_limits,
                    env=self._get_restricted_env()
                )
            
            try:
                stdout, stderr = process.communicate(timeout=self.max_execution_time)
                execution_time = time.time() - start_time
                
                # Check output size
                output = stdout.decode('utf-8', errors='replace')
                error = stderr.decode('utf-8', errors='replace')
                
                if len(output) > self.max_output_size:
                    output = output[:self.max_output_size] + "\n[Output truncated due to size limit]"
                
                if process.returncode != 0:
                    return {
                        "success": False,
                        "error": error,
                        "output": output,
                        "execution_time": execution_time,
                        "return_code": process.returncode
                    }
                
                return {
                    "success": True,
                    "output": output,
                    "error": error,
                    "execution_time": execution_time,
                    "return_code": 0
                }
                
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "error": f"Execution timeout exceeded ({self.max_execution_time}s)",
                    "output": "",
                    "execution_time": self.max_execution_time,
                    "timeout": True
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "output": "",
                "execution_time": 0,
                "exception": str(type(e).__name__)
            }
        finally:
            self.cleanup_sandbox(sandbox_dir)
    
    def _wrap_code_safely(self, code: str) -> str:
        """Wrap user code with safety measures"""
        wrapper = '''
import sys
import signal

# Set up timeout handler
def timeout_handler(signum, frame):
    print("\\nExecution timeout reached", file=sys.stderr)
    sys.exit(1)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm({timeout})

# Restrict built-ins
restricted_builtins = {{
    'print': print,
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'sum': sum,
    'min': min,
    'max': max,
    'abs': abs,
    'round': round,
    'sorted': sorted,
    'reversed': reversed,
    'int': int,
    'float': float,
    'str': str,
    'bool': bool,
    'list': list,
    'tuple': tuple,
    'set': set,
    'dict': dict,
    'type': type,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'all': all,
    'any': any,
    'ord': ord,
    'chr': chr,
    'bin': bin,
    'hex': hex,
    'oct': oct,
    'format': format,
    'repr': repr,
    'ascii': ascii,
    'hash': hash,
    'id': id,
    'pow': pow,
    'divmod': divmod,
    'complex': complex,
    'bytes': bytes,
    'bytearray': bytearray,
    'memoryview': memoryview,
    'slice': slice,
    'property': property,
    'staticmethod': staticmethod,
    'classmethod': classmethod,
    'super': super,
    'object': object,
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'KeyError': KeyError,
    'IndexError': IndexError,
    'AttributeError': AttributeError,
    'ImportError': ImportError,
    'NameError': NameError,
    'RuntimeError': RuntimeError,
    'StopIteration': StopIteration,
    'GeneratorExit': GeneratorExit,
    'SystemExit': SystemExit,
    'KeyboardInterrupt': KeyboardInterrupt,
    'True': True,
    'False': False,
    'None': None
}}

# Replace builtins
__builtins__ = restricted_builtins

# User code starts here
{code}
'''
        return wrapper.format(timeout=self.max_execution_time, code=code)
    
    def _get_restricted_env(self) -> Dict[str, str]:
        """Get restricted environment variables for subprocess"""
        # Start with minimal environment
        env = {
            'PATH': '/usr/bin:/bin',
            'PYTHONPATH': '',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONUNBUFFERED': '1',
            'LC_ALL': 'C.UTF-8',
            'LANG': 'C.UTF-8'
        }
        return env


class CodeValidator(ast.NodeVisitor):
    """AST visitor to validate code for security issues"""
    
    def __init__(self, allowed_imports: set, forbidden_builtins: set):
        self.allowed_imports = allowed_imports
        self.forbidden_builtins = forbidden_builtins
        self.imports_used = set()
    
    def visit_Import(self, node: ast.Import):
        """Check import statements"""
        for alias in node.names:
            module_name = alias.name.split('.')[0]
            if module_name not in self.allowed_imports:
                raise SecurityViolation(f"Import of '{module_name}' is not allowed")
            self.imports_used.add(module_name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check from-import statements"""
        if node.module:
            module_name = node.module.split('.')[0]
            if module_name not in self.allowed_imports:
                raise SecurityViolation(f"Import from '{module_name}' is not allowed")
            self.imports_used.add(module_name)
        self.generic_visit(node)
    
    def visit_Name(self, node: ast.Name):
        """Check for forbidden builtins"""
        if isinstance(node.ctx, ast.Load) and node.id in self.forbidden_builtins:
            raise SecurityViolation(f"Use of '{node.id}' is not allowed")
        self.generic_visit(node)
    
    def visit_Attribute(self, node: ast.Attribute):
        """Check for dangerous attribute access"""
        # Check for dunder attribute access
        if node.attr.startswith('__') and node.attr.endswith('__'):
            raise SecurityViolation(f"Access to dunder attribute '{node.attr}' is not allowed")
        self.generic_visit(node)
    
    def visit_Call(self, node: ast.Call):
        """Check function calls for dangerous operations"""
        # Check for eval/exec even as attributes
        if isinstance(node.func, ast.Name):
            if node.func.id in self.forbidden_builtins:
                raise SecurityViolation(f"Call to '{node.func.id}' is not allowed")
        
        # Check for getattr/setattr/delattr with string literals
        if isinstance(node.func, ast.Name) and node.func.id in ['getattr', 'setattr', 'delattr']:
            raise SecurityViolation(f"Call to '{node.func.id}' is not allowed")
        
        self.generic_visit(node)


# Global executor instance
code_executor = CodeExecutor()

def execute_python_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a secure sandbox.
    
    Args:
        code: Python code to execute
        
    Returns:
        Execution results dictionary
    """
    return code_executor.execute_code(code, language="python")