correct_rates = [0.3, 0.5, 0.7, 0.88, 0.94, 1.00]
length_multipliers = [1, 2, 3, 4, 5, 6]

best_efficiency = 0
best_multiplier = None

for rate, multiplier in zip(correct_rates, length_multipliers):
    error_rate = 1 - rate
    ecc_bytes_needed_per_byte = error_rate * 2 * (1 / multiplier)
    total_bytes_per_byte = 1 / multiplier + ecc_bytes_needed_per_byte
    # 计算有效率
    efficiency = multiplier / total_bytes_per_byte

    if efficiency > best_efficiency:
        best_efficiency = efficiency
        best_multiplier = multiplier

print(f"The best strategy is to use a length multiplier of {best_multiplier} for highest efficiency.")
