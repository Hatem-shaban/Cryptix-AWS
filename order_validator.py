"""
Order validation utilities for preventing Binance API filter errors
"""

import logging
from typing import Dict, Any, Tuple, Optional
from decimal import Decimal, ROUND_DOWN


class OrderValidator:
    """Helper class to validate orders before placing them on Binance"""
    
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__)
    
    def get_symbol_filters(self, symbol: str) -> Dict[str, Any]:
        """Get symbol filters from exchange info"""
        try:
            # Assuming get_exchange_info_cached exists in the main module
            from web_bot import get_exchange_info_cached
            
            exchange_info = get_exchange_info_cached()
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol), None)
            
            if not symbol_info:
                raise ValueError(f"Symbol {symbol} not found in exchange info")
            
            filters = {}
            for filter_info in symbol_info['filters']:
                filters[filter_info['filterType']] = filter_info
            
            return filters
            
        except Exception as e:
            self.logger.error(f"Error getting symbol filters for {symbol}: {e}")
            raise
    
    def validate_lot_size(self, symbol: str, quantity: float) -> Tuple[float, bool]:
        """
        Validate and adjust quantity according to LOT_SIZE filter
        
        Returns:
            Tuple[float, bool]: (adjusted_quantity, is_valid)
        """
        try:
            filters = self.get_symbol_filters(symbol)
            lot_size_filter = filters.get('LOT_SIZE')
            
            if not lot_size_filter:
                return quantity, True
            
            min_qty = float(lot_size_filter['minQty'])
            max_qty = float(lot_size_filter['maxQty'])
            step_size = float(lot_size_filter['stepSize'])
            
            # Check minimum quantity
            if quantity < min_qty:
                self.logger.warning(f"Quantity {quantity:.8f} below minimum {min_qty:.8f} for {symbol}")
                return min_qty, quantity >= min_qty
            
            # Check maximum quantity
            if quantity > max_qty:
                self.logger.warning(f"Quantity {quantity:.8f} exceeds maximum {max_qty:.8f} for {symbol}")
                return max_qty, False
            
            # Adjust to step size using Decimal for precision
            decimal_qty = Decimal(str(quantity))
            decimal_step = Decimal(str(step_size))
            decimal_min = Decimal(str(min_qty))
            
            # Calculate steps from minimum
            steps_from_min = ((decimal_qty - decimal_min) / decimal_step).quantize(Decimal('1'), rounding=ROUND_DOWN)
            adjusted_qty = float(decimal_min + (steps_from_min * decimal_step))
            
            # Final check
            if adjusted_qty < min_qty:
                adjusted_qty = min_qty
            
            return adjusted_qty, True
            
        except Exception as e:
            self.logger.error(f"Error validating lot size for {symbol}: {e}")
            return quantity, False
    
    def validate_notional_value(self, symbol: str, quantity: float, price: float = None) -> Tuple[bool, float]:
        """
        Validate that the order meets MIN_NOTIONAL requirements
        
        Returns:
            Tuple[bool, float]: (is_valid, min_notional_required)
        """
        try:
            filters = self.get_symbol_filters(symbol)
            min_notional_filter = filters.get('MIN_NOTIONAL')
            
            if not min_notional_filter:
                return True, 0.0
            
            min_notional = float(min_notional_filter['minNotional'])
            
            # Get current price if not provided
            if price is None:
                ticker = self.client.get_ticker(symbol=symbol)
                price = float(ticker['lastPrice'])
            
            notional_value = quantity * price
            is_valid = notional_value >= min_notional
            
            if not is_valid:
                self.logger.warning(
                    f"Notional value {notional_value:.2f} below minimum {min_notional:.2f} for {symbol}"
                )
            
            return is_valid, min_notional
            
        except Exception as e:
            self.logger.error(f"Error validating notional value for {symbol}: {e}")
            return False, 10.0  # Conservative fallback
    
    def calculate_minimum_valid_quantity(self, symbol: str, price: float = None) -> float:
        """
        Calculate the minimum valid quantity that satisfies both LOT_SIZE and MIN_NOTIONAL
        """
        try:
            # Get current price if not provided
            if price is None:
                ticker = self.client.get_ticker(symbol=symbol)
                price = float(ticker['lastPrice'])
            
            filters = self.get_symbol_filters(symbol)
            
            # Get minimum from LOT_SIZE
            lot_size_filter = filters.get('LOT_SIZE', {})
            min_qty_lot = float(lot_size_filter.get('minQty', 0.001))
            step_size = float(lot_size_filter.get('stepSize', 0.001))
            
            # Get minimum from MIN_NOTIONAL
            min_notional_filter = filters.get('MIN_NOTIONAL', {})
            min_notional = float(min_notional_filter.get('minNotional', 10.0))
            min_qty_notional = min_notional / price
            
            # Take the maximum to satisfy both constraints
            min_qty = max(min_qty_lot, min_qty_notional)
            
            # Adjust to step size properly - round UP to ensure we meet minimum notional
            from decimal import Decimal, ROUND_UP
            decimal_qty = Decimal(str(min_qty))
            decimal_step = Decimal(str(step_size))
            decimal_min = Decimal(str(min_qty_lot))
            
            # Calculate steps needed from minimum, rounding UP
            if decimal_qty <= decimal_min:
                adjusted_qty = float(decimal_min)
            else:
                steps_from_min = ((decimal_qty - decimal_min) / decimal_step).quantize(Decimal('1'), rounding=ROUND_UP)
                adjusted_qty = float(decimal_min + (steps_from_min * decimal_step))
            
            # Final verification that result meets both constraints
            final_notional = adjusted_qty * price
            if final_notional < min_notional:
                # Need to add one more step
                adjusted_qty += step_size
                final_notional = adjusted_qty * price
                
            # Log calculation details for debugging
            self.logger.debug(f"Minimum quantity calculation for {symbol}:")
            self.logger.debug(f"  Price: ${price:.4f}")
            self.logger.debug(f"  Min LOT_SIZE qty: {min_qty_lot:.8f}")
            self.logger.debug(f"  Min NOTIONAL qty: {min_qty_notional:.8f} (for ${min_notional:.2f})")
            self.logger.debug(f"  Step size: {step_size:.8f}")
            self.logger.debug(f"  Calculated minimum: {adjusted_qty:.8f}")
            self.logger.debug(f"  Final notional: ${final_notional:.2f}")
            
            return adjusted_qty
            
        except Exception as e:
            self.logger.error(f"Error calculating minimum valid quantity for {symbol}: {e}")
            # Return a conservative fallback that should work for most symbols
            return 10.0 / (price or 1.0)  # $10 worth at current price
    
    def validate_order(self, symbol: str, quantity: float, side: str = 'BUY', price: float = None, available_balance: float = None) -> Dict[str, Any]:
        """
        Comprehensive order validation including balance checks
        
        Args:
            available_balance: For SELL orders, the actual available balance of the base asset
        
        Returns:
            Dict with validation results and adjusted parameters
        """
        result = {
            'is_valid': False,
            'original_quantity': quantity,
            'adjusted_quantity': quantity,
            'errors': [],
            'warnings': [],
            'min_notional_required': 0.0,
            'current_price': price,
            'available_balance': available_balance
        }
        
        try:
            # Get current price if needed
            if price is None:
                ticker = self.client.get_ticker(symbol=symbol)
                result['current_price'] = float(ticker['lastPrice'])
                price = result['current_price']
            
            # EARLY CHECK: Validate minimum notional FIRST
            notional_valid, min_notional = self.validate_notional_value(symbol, quantity, price)
            result['min_notional_required'] = min_notional
            
            if not notional_valid:
                current_notional = quantity * price
                self.logger.warning(f"Order fails minimum notional: ${current_notional:.2f} < ${min_notional:.2f}")
                
                # Calculate minimum quantity needed for notional BEFORE other validations
                min_valid_qty = self.calculate_minimum_valid_quantity(symbol, price)
                
                # For SELL orders, check if we have enough balance for minimum notional
                if side.upper() == 'SELL' and available_balance is not None:
                    if min_valid_qty > available_balance:
                        max_possible_notional = available_balance * price
                        result['errors'].append(
                            f"Balance insufficient for minimum notional. Have ${max_possible_notional:.2f}, need ${min_notional:.2f}"
                        )
                        return result
                    else:
                        # We have enough balance, adjust quantity to minimum valid
                        result['adjusted_quantity'] = min_valid_qty
                        result['warnings'].append(f"Quantity increased to meet minimum notional: {min_valid_qty:.8f}")
                        quantity = min_valid_qty
                else:
                    # BUY order or SELL without balance check - adjust to minimum
                    result['adjusted_quantity'] = min_valid_qty
                    result['warnings'].append(f"Quantity increased to meet minimum notional: {min_valid_qty:.8f}")
                    quantity = min_valid_qty
            
            # For SELL orders, check available balance after notional adjustment
            if side.upper() == 'SELL' and available_balance is not None:
                if quantity > available_balance:
                    result['errors'].append(
                        f"Adjusted quantity {quantity:.8f} exceeds available balance {available_balance:.8f}"
                    )
                    # Try using available balance if it meets notional
                    balance_notional = available_balance * price
                    if balance_notional >= min_notional:
                        result['adjusted_quantity'] = available_balance
                        result['warnings'].append(f"Quantity reduced to available balance: {available_balance:.8f}")
                        quantity = available_balance
                    else:
                        result['errors'].append(
                            f"Available balance too small for minimum notional: ${balance_notional:.2f} < ${min_notional:.2f}"
                        )
                        return result
            
            # Now validate LOT_SIZE with the notional-adjusted quantity
            adjusted_qty, lot_valid = self.validate_lot_size(symbol, quantity)
            result['adjusted_quantity'] = adjusted_qty
            
            if not lot_valid:
                result['errors'].append(f"LOT_SIZE validation failed for quantity {quantity:.8f}")
            elif adjusted_qty != quantity:
                result['warnings'].append(f"Quantity adjusted from {quantity:.8f} to {adjusted_qty:.8f} for LOT_SIZE compliance")
                
                # RE-CHECK notional after LOT_SIZE adjustment
                final_notional = adjusted_qty * price
                if final_notional < min_notional:
                    result['errors'].append(
                        f"After LOT_SIZE adjustment, below minimum notional: ${final_notional:.2f} < ${min_notional:.2f}"
                    )
                    return result
            
            # Final balance check after all adjustments
            if side.upper() == 'SELL' and available_balance is not None:
                if adjusted_qty > available_balance:
                    result['errors'].append(
                        f"Final adjusted quantity {adjusted_qty:.8f} exceeds available balance {available_balance:.8f}"
                    )
                    return result
            
            # If we get here, all validations passed
            result['is_valid'] = True
            
            return result
            
        except Exception as e:
            result['errors'].append(f"Order validation error: {str(e)}")
            self.logger.error(f"Order validation failed for {symbol}: {e}")
            return result


def log_validation_result(result: Dict[str, Any], symbol: str, function_name: str = "order_validation"):
    """Log validation results to CSV if needed"""
    try:
        if result['errors']:
            from web_bot import log_error_to_csv
            errors_str = '; '.join(result['errors'])
            log_error_to_csv(f"Order validation failed for {symbol}: {errors_str}", 
                           "VALIDATION_ERROR", function_name, "ERROR")
        
        if result['warnings']:
            from web_bot import log_error_to_csv
            warnings_str = '; '.join(result['warnings'])
            log_error_to_csv(f"Order validation warnings for {symbol}: {warnings_str}", 
                           "VALIDATION_WARNING", function_name, "WARNING")
    except Exception as e:
        print(f"Error logging validation result: {e}")


def validate_order_before_execution(client, symbol: str, quantity: float, side: str = 'BUY') -> Tuple[bool, float, str]:
    """
    Quick validation function that can be imported and used anywhere
    
    Returns:
        Tuple[bool, float, str]: (is_valid, adjusted_quantity, error_message)
    """
    try:
        validator = OrderValidator(client)
        
        # Get current price for validation
        current_price = float(client.get_ticker(symbol=symbol)['lastPrice'])
        
        # For SELL orders, get available balance
        available_balance = None
        if side.upper() == 'SELL':
            base_asset = symbol.replace('USDT', '').replace('BTC', '').replace('ETH', '').replace('BNB', '')
            if symbol.endswith('USDT'):
                base_asset = symbol[:-4]
            elif symbol.endswith('BTC'):
                base_asset = symbol[:-3]
            
            account_info = client.get_account()
            for balance in account_info['balances']:
                if balance['asset'] == base_asset:
                    available_balance = float(balance['free'])
                    break
        
        result = validator.validate_order(symbol, quantity, side, current_price, available_balance)
        
        if result['is_valid']:
            return True, result['adjusted_quantity'], ""
        else:
            error_msg = '; '.join(result['errors'])
            return False, quantity, error_msg
            
    except Exception as e:
        return False, quantity, f"Validation error: {str(e)}"
