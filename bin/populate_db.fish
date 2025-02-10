#!/usr/bin/env fish

echo "Attempting to populate the database in the 'svm' service..."
docker-compose exec svm flask populate_db

if test $status -eq 0
    echo "Database population command executed successfully."
else
    echo "Error executing database population command. Exit status: $status"
end
