package com.service.accountservice.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.validation.constraints.NotNull;
import lombok.RequiredArgsConstructor;
import org.springframework.lang.NonNull;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.web.authentication.WebAuthenticationDetailsSource;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import java.io.IOException;

@Component
@RequiredArgsConstructor
public class JwtAuthenticationFilter extends OncePerRequestFilter {

    private final JwtService jwtService;
    private final UserDetailsService userDetailsService;

    @Override
    protected void doFilterInternal(
            @NonNull HttpServletRequest request,
            @NonNull HttpServletResponse response,
            @NonNull FilterChain filterChain)
            throws ServletException, IOException {
        final String authHeader = request.getHeader("Authorization");
        final String jwt;
        final String userEmail;
        // If token is not present, or authentication is not JWT type, then pass the request to the next filter
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            filterChain.doFilter(request, response);
            return;
        }
        // If token is present, then we extract the token from the HTTP request's header
        jwt = authHeader.substring(7);
        // Get the email embedded in the token using JWTService that you have implemented
        userEmail = jwtService.extractUsername(jwt);
        /* If the email is not null, and the user is not already authenticated, then we authenticate the user
        We do this when the second request sent, after the /authenticate endpoint is called
        Because the authenticate endpoint merely creates the token, it does not actually set the user
        in the securityContext */
        if(userEmail != null && SecurityContextHolder.getContext().getAuthentication() == null){
            // Here we are loading the user details from the DB by querying the DB with the username/email in the token
            UserDetails userDetails = this.userDetailsService.loadUserByUsername(userEmail);

            /*This method checks if the username in the token actually returns a user from the DB, and if the usernames equate.
            If it does not, means the user does not exist.
            If it does, we can proceed to authenticate the user */
            if(jwtService.isTokenValid(jwt,userDetails)){
                UsernamePasswordAuthenticationToken usernamePasswordAuthenticationToken = new UsernamePasswordAuthenticationToken(
                        userDetails,null,userDetails.getAuthorities()
                );
                usernamePasswordAuthenticationToken.setDetails(new WebAuthenticationDetailsSource().buildDetails(request));
                // Authenticating the user involves setting the user in the securityContext
                SecurityContextHolder.getContext().setAuthentication(usernamePasswordAuthenticationToken);
            }
        }
        filterChain.doFilter(request,response);
    }
}
