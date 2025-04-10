import tkinter as tk
from tkinter import ttk
import json
from datetime import datetime, timedelta

class StrategyConfig:
    def __init__(self):
        self.initial_capital = 100000
        self.risk_level = "Moderate"  # Conservative, Moderate, Aggressive
        self.position_size = 0.1  # Percentage of capital per trade
        self.stop_loss = 0.02  # 2% stop loss
        self.take_profit = 0.03  # 3% take profit
        self.max_positions = 1  # Maximum number of concurrent positions
        self.trading_hours = "Regular"  # Regular, Extended, 24/7
        self.commission = 0.0001  # 0.01% commission per trade
        self.leverage = 1  # No leverage by default
        self.backtest_days = 5
        self.start_date = None
        self.end_date = None
        
    def to_dict(self):
        return {
            'initial_capital': self.initial_capital,
            'risk_level': self.risk_level,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_positions': self.max_positions,
            'trading_hours': self.trading_hours,
            'commission': self.commission,
            'leverage': self.leverage,
            'backtest_days': self.backtest_days,
            'start_date': self.start_date,
            'end_date': self.end_date
        }
    
    def from_dict(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class StrategyConfigGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Trading Strategy Configuration")
        self.config = StrategyConfig()
        self.setup_gui()
        
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Capital and Risk Settings
        ttk.Label(main_frame, text="Capital and Risk Settings", font=('Helvetica', 12, 'bold')).grid(row=0, column=0, columnspan=2, pady=10)
        
        # Initial Capital
        ttk.Label(main_frame, text="Initial Capital ($):").grid(row=1, column=0, sticky=tk.W)
        self.capital_var = tk.StringVar(value=str(self.config.initial_capital))
        ttk.Entry(main_frame, textvariable=self.capital_var).grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Risk Level
        ttk.Label(main_frame, text="Risk Level:").grid(row=2, column=0, sticky=tk.W)
        self.risk_var = tk.StringVar(value=self.config.risk_level)
        risk_combo = ttk.Combobox(main_frame, textvariable=self.risk_var, values=["Conservative", "Moderate", "Aggressive"])
        risk_combo.grid(row=2, column=1, sticky=(tk.W, tk.E))
        
        # Position Sizing
        ttk.Label(main_frame, text="Position Size (%):").grid(row=3, column=0, sticky=tk.W)
        self.position_var = tk.StringVar(value=str(self.config.position_size * 100))
        ttk.Entry(main_frame, textvariable=self.position_var).grid(row=3, column=1, sticky=(tk.W, tk.E))
        
        # Stop Loss and Take Profit
        ttk.Label(main_frame, text="Stop Loss (%):").grid(row=4, column=0, sticky=tk.W)
        self.sl_var = tk.StringVar(value=str(self.config.stop_loss * 100))
        ttk.Entry(main_frame, textvariable=self.sl_var).grid(row=4, column=1, sticky=(tk.W, tk.E))
        
        ttk.Label(main_frame, text="Take Profit (%):").grid(row=5, column=0, sticky=tk.W)
        self.tp_var = tk.StringVar(value=str(self.config.take_profit * 100))
        ttk.Entry(main_frame, textvariable=self.tp_var).grid(row=5, column=1, sticky=(tk.W, tk.E))
        
        # Trading Settings
        ttk.Label(main_frame, text="Trading Settings", font=('Helvetica', 12, 'bold')).grid(row=6, column=0, columnspan=2, pady=10)
        
        # Max Positions
        ttk.Label(main_frame, text="Max Concurrent Positions:").grid(row=7, column=0, sticky=tk.W)
        self.max_pos_var = tk.StringVar(value=str(self.config.max_positions))
        ttk.Entry(main_frame, textvariable=self.max_pos_var).grid(row=7, column=1, sticky=(tk.W, tk.E))
        
        # Trading Hours
        ttk.Label(main_frame, text="Trading Hours:").grid(row=8, column=0, sticky=tk.W)
        self.hours_var = tk.StringVar(value=self.config.trading_hours)
        hours_combo = ttk.Combobox(main_frame, textvariable=self.hours_var, values=["Regular", "Extended", "24/7"])
        hours_combo.grid(row=8, column=1, sticky=(tk.W, tk.E))
        
        # Commission
        ttk.Label(main_frame, text="Commission (%):").grid(row=9, column=0, sticky=tk.W)
        self.commission_var = tk.StringVar(value=str(self.config.commission * 100))
        ttk.Entry(main_frame, textvariable=self.commission_var).grid(row=9, column=1, sticky=(tk.W, tk.E))
        
        # Leverage
        ttk.Label(main_frame, text="Leverage:").grid(row=10, column=0, sticky=tk.W)
        self.leverage_var = tk.StringVar(value=str(self.config.leverage))
        ttk.Entry(main_frame, textvariable=self.leverage_var).grid(row=10, column=1, sticky=(tk.W, tk.E))
        
        # Backtest Period
        ttk.Label(main_frame, text="Backtest Period (days):").grid(row=11, column=0, sticky=tk.W)
        self.period_var = tk.StringVar(value=str(self.config.backtest_days))
        ttk.Entry(main_frame, textvariable=self.period_var).grid(row=11, column=1, sticky=(tk.W, tk.E))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=12, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="Run Backtest", command=self.run_backtest).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=5)
        
    def update_config(self):
        try:
            self.config.initial_capital = float(self.capital_var.get())
            self.config.risk_level = self.risk_var.get()
            self.config.position_size = float(self.position_var.get()) / 100
            self.config.stop_loss = float(self.sl_var.get()) / 100
            self.config.take_profit = float(self.tp_var.get()) / 100
            self.config.max_positions = int(self.max_pos_var.get())
            self.config.trading_hours = self.hours_var.get()
            self.config.commission = float(self.commission_var.get()) / 100
            self.config.leverage = float(self.leverage_var.get())
            self.config.backtest_days = int(self.period_var.get())
            return True
        except ValueError as e:
            tk.messagebox.showerror("Error", f"Invalid input: {str(e)}")
            return False
    
    def run_backtest(self):
        if self.update_config():
            self.root.destroy()
            # Import and run backtest with config
            from backtest_strategy import backtest_strategy
            backtest_strategy(self.config)
    
    def save_config(self):
        if self.update_config():
            filename = f"strategy_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=4)
            tk.messagebox.showinfo("Success", f"Configuration saved to {filename}")
    
    def load_config(self):
        filename = tk.filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_dict = json.load(f)
                self.config.from_dict(config_dict)
                self.update_gui_from_config()
                tk.messagebox.showinfo("Success", "Configuration loaded successfully")
            except Exception as e:
                tk.messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def update_gui_from_config(self):
        self.capital_var.set(str(self.config.initial_capital))
        self.risk_var.set(self.config.risk_level)
        self.position_var.set(str(self.config.position_size * 100))
        self.sl_var.set(str(self.config.stop_loss * 100))
        self.tp_var.set(str(self.config.take_profit * 100))
        self.max_pos_var.set(str(self.config.max_positions))
        self.hours_var.set(self.config.trading_hours)
        self.commission_var.set(str(self.config.commission * 100))
        self.leverage_var.set(str(self.config.leverage))
        self.period_var.set(str(self.config.backtest_days))
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = StrategyConfigGUI()
    gui.run() 