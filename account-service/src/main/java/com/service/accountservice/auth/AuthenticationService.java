package com.service.accountservice.auth;

import com.service.accountservice.config.JwtService;
import com.service.accountservice.model.Account;
import com.service.accountservice.repository.AccountServiceRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;


@Service
@RequiredArgsConstructor
public class AuthenticationService {
    private final AuthenticationManager authenticationManager;
    private final AccountServiceRepository repository;
    private final JwtService jwtService;
    private final PasswordEncoder passwordEncoder;

    public AuthenticationResponse authenticate(AuthenticationRequest request) {
        authenticationManager.authenticate(
                new UsernamePasswordAuthenticationToken(
                        request.getEmail(),
                        request.getPassword()
                )
        );
        var account = repository.findByEmail(request.getEmail());
        var jwtToken = jwtService.generateToken(account);
        return AuthenticationResponse.builder()
                .accessToken(jwtToken)
                .build();
    }

    public AuthenticationResponse register(RegisterRequest request) {
        var account = Account.builder()
                .name(request.getName())
                .email(request.getEmail())
                .password(passwordEncoder.encode(request.getPassword()))
                .role(request.getRole())
                .build();
        var savedAccount = repository.save(account);
        var jwtToken = jwtService.generateToken(savedAccount);
        return AuthenticationResponse.builder()
                .accessToken(jwtToken)
                .build();
    }





}
