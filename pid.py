class PID:
    def __init__(self):
        """Constructor"""

        """
        Errors
        """
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

        """
        Coefficients
        """
        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

    def init(self, Kp, Ki, Kd):
        """Initialize PID."""
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def update_error(self, cte):
        """Update the PID error variables given cross track error."""
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte

    def total_error(self):
        """Calculate the total PID error."""
        return 0-(self.Kp*self.p_error + self.Ki*self.i_error + self.Kd*self.d_error)

class TwiddlingPID(PID):
    def __init__(self, init_dp, max_cte, n):
        self._dp = init_dp
        self._max_cte = max_cte
        self._n = n
        self._tuning_idx = 0
        self._increment = True
        self._error = 0
        self._i = 0
        self.best_error = 10 * max_cte*max_cte

    def update_error(self, cte):
        super(TwiddlingPID, self).update_error(cte)
        self._i = self._i+1
        if self._i > self._n:
            self._error += cte* cte

    def total_error(self):
        pid_error = super(TwiddlingPID, self).total_error()

        dp_sum = sum(self._dp)
        if(dp_sum < 0.001): #Twiddling finished. Continue;
            return pid_error
        if (abs(self.p_error) > self._max_cte and self._i > 10): #Crashed
            pid_error = None
            self._twiddle()
        elif (self._i > 2*self._n):
            pid_error = None
            self._twiddle()

        return pid_error

    def _twiddle(self):
        err = 0.0
        if (self._i <= self.n) :
            err = self._max_cte*self._max_cte
        else:
            err = self._error*self._n/(self._i*(self._i-self._n))

        if self._increment:
            if err < self.best_error:
                self.best_error = err
                self._dp[self._tuning_idx] *= 1.1
                self._next_coefficient()
                self._update_coefficient(self._dp[self._tuning_idx])
            else:
                self._update_coefficient(-2*self._dp[self._tuning_idx])
                self._increment = False
        else:
            if err < self.best_error:
                self.best_error = err
                self._dp[self._tuning_idx] *= 1.1
            else:
                self._update_coefficient(self._dp[self._tuning_idx])
                self._dp[self._tuning_idx] *= 0.9
            self._next_coefficient()
            self._update_coefficient(self._dp[self._tuning_idx])
            self._increment = True
        self._error = 0.0
        self._i = 0
        self.p_error = 0
        self.i_error = 0
        self.d_error = 0

    def _update_coefficient(self, dp):
        if self._tuning_idx == 0:
            self.Kp += dp
        if self._tuning_idx == 1:
            self.Ki += dp
        if self._tuning_idx == 2:
            self.Kd += dp

    def _next_coefficient(self):
        dp_sum = sum(self._dp)
        if(dp_sum > 0.001):
            first_pass = True
            while first_pass or self._dp[self._tuning_idx] < 0.0001:
                first_pass = False
                self._tuning_idx = (self._tuning_idx + 1) % 3


