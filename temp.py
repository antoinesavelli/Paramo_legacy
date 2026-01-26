import subprocess
import shutil

# Method 1: Check drive type with PowerShell
print("=== T: Drive Type ===")
result = subprocess.run(
    ['powershell', '-Command', 'Get-PSDrive T | Select-Object Name,Provider,Root,Used,Free'],
    capture_output=True, text=True
)
print(result.stdout)

# Method 2: Check disk info
print("\n=== Disk Info ===")
result = subprocess.run(
    ['powershell', '-Command', 'Get-PhysicalDisk | Select-Object FriendlyName,MediaType,BusType,Size'],
    capture_output=True, text=True
)
print(result.stdout)

# Method 3: Simple space check
print("\n=== T: Drive Space ===")
total, used, free = shutil.disk_usage('T:\\')
print(f"Total: {total / (1024**3):.1f} GB")
print(f"Used: {used / (1024**3):.1f} GB")
print(f"Free: {free / (1024**3):.1f} GB")

# Method 4: Check cluster size
print("\n=== Cluster Size ===")
result = subprocess.run(['fsutil', 'fsinfo', 'ntfsinfo', 'T:'], 
                        capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if 'Bytes Per Cluster' in line:
        print(line.strip())