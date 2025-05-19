import winreg
import wmi

def get_friendly_names():
    """
    Get a mapping of GUID to friendly interface names using WMI.
    """
    friendly_names = {}
    c = wmi.WMI()
    for nic in c.Win32_NetworkAdapterConfiguration(IPEnabled=True):
        guid = nic.SettingID
        name = nic.Description
        if guid:
            friendly_names[guid.upper()] = name
    return friendly_names

def get_npf_devices():
    """
    Get all installed NPF interfaces from the registry.
    """
    devices = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"SYSTEM\CurrentControlSet\Services\Npcap\Interfaces")
        i = 0
        while True:
            try:
                guid = winreg.EnumKey(key, i)
                devices.append(guid.upper())
                i += 1
            except OSError:
                break
    except FileNotFoundError:
        print("[!] Npcap not installed or missing registry key.")
    return devices

def main():
    npf_devices = get_npf_devices()
    friendly_names = get_friendly_names()

    if not npf_devices:
        print("[!] No NPF interfaces found.")
        return

    print("Available NPF Interfaces:\n")
    for guid in npf_devices:
        device_path = f"\\Device\\NPF_{{{guid}}}"
        name = friendly_names.get(guid, "(Unknown / Unused interface)")
        print(f"{device_path:<60} => {name}")

if __name__ == "__main__":
    main()
